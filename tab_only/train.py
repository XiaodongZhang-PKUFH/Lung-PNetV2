import warnings
warnings.filterwarnings('ignore')

from typing import Tuple, List
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from fastai.tabular.all import *
from config import *

def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess labels to 'IAC'/'Other' categories"""
    df['Label2'] = df['Label'].apply(lambda x: 'Other' if x == 1 else 'IAC')
    return df

def create_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Create stratified KFold cross-validation splits"""
    df['fold'] = -1
    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    
    for i, (_, valid_index) in enumerate(strat_kfold.split(df.PatientID.values, df['Label2'].values)):
        df.iloc[valid_index, -1] = i
    
    df['fold'] = df['fold'].astype('int')
    return df

def get_tabular_data(df: pd.DataFrame, fold: int) -> TabularPandas:
    """Get tabular data for specified fold"""
    df.loc[df.fold==fold, 'Cat'] = 'val'
    df['Cat'] = df['Cat'].apply(lambda x: 1 if x =='val' else 0)
    
    return TabularPandas(
        df, [Categorify, FillMissing, Normalize],
        CAT_NAMES, CONT_NAMES,
        y_names='Label', y_block=CategoryBlock(),
        splits=(df[df.fold!=fold].index, df[df.fold==fold].index)
    )

def train_fold(to: TabularPandas, fold: int) -> Learner:
    """Train model for specified fold"""
    tab_dl = to.dataloaders(bs=BS)
    emb_szs = get_emb_sz(tab_dl)
    tab_model = TabularModel(emb_szs, len(CONT_NAMES), out_sz=get_c(tab_dl), layers=[128], ps=0.2)
    
    learn = Learner(
        tab_dl, 
        tab_model,
        metrics=[accuracy, RocAucBinary()],
        opt_func=ranger,
        loss_func=CrossEntropyLossFlat(),
        model_dir=MODEL_DIR,
        cbs=[
            Recorder(train_metrics=True),
            SaveModelCallback(
                monitor='valid_roc_auc_score',
                with_opt=True,
                fname=MODEL_NAME_TEMPLATE.format(fold=fold)
            ),
            CSVLogger(fname=MODEL_DIR/HISTORY_LOG_TEMPLATE.format(fold=fold)),
            ReduceLROnPlateau(patience=5, monitor='valid_roc_auc_score', factor=50),
            EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=100)
        ]
    )
    
    learn.to_fp16()
    learn.fit_flat_cos(10, INIT_LR)
    learn.unfreeze()
    learn.fit_flat_cos(NUM_EPOCH, lr=slice(3e-6, 3e-4))
    return learn

def evaluate_test(learn: Learner, test_df: pd.DataFrame) -> None:
    """Evaluate model on test set"""
    test_df['Cat'] = 'test'
    to_test = TabularPandas(
        test_df, [Categorify, FillMissing, Normalize],
        CAT_NAMES, CONT_NAMES,
        y_names='Label', y_block=CategoryBlock(),
        splits=ColSplitter('Cat')(test_df)
    )
    tab_dl_test = to_test.dataloaders(bs=1)
    
    learn_test = Learner(
        tab_dl_test,
        learn.model,
        metrics=[accuracy, RocAucBinary()],
        opt_func=ranger,
        loss_func=CrossEntropyLossFlat(),
        model_dir=MODEL_DIR
    )
    learn_test.to_fp16()
    learn_test.load(MODEL_LOAD_TEMPLATE.format(fold=learn.cbs[1].fname.split('_')[-1]))
    return learn_test.get_preds(ds_idx=1)

def main():
    # Load and preprocess data
    train_df = preprocess_labels(pd.read_csv(PROCESSED_DATA/TRAIN_CSV))
    test_df = preprocess_labels(pd.read_csv(PROCESSED_DATA/TEST_CSV))
    train_df = create_folds(train_df)
    
    # Train each fold
    for fold in range(N_FOLDS):
        print(f"Training fold {fold}")
        to = get_tabular_data(train_df, fold)
        learn = train_fold(to, fold)
        preds, y = evaluate_test(learn, test_df)

if __name__ == '__main__':
    main()
