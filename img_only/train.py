import warnings
warnings.filterwarnings('ignore')

from typing import Tuple
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from faimed3d.all import *
from fastai.callback.all import *
import torch

from config import *

def preprocess_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess labels by converting to 'IAC'/'Other' categories."""
    df['Label2'] = df['Label'].apply(lambda x: 'Other' if x == 1 else 'IAC')
    return df

def setup_kfold(df: pd.DataFrame) -> pd.DataFrame:
    """Setup KFold cross-validation splits."""
    df['fold'] = -1
    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    
    for i, (_, valid_index) in enumerate(strat_kfold.split(df.PatientID.values, df['Label2'].values)):
        df.iloc[valid_index, -1] = i
    
    df['fold'] = df['fold'].astype('int')
    df.loc[df.loc[df.fold==FOLD].index, 'Cat'] = 'val'
    df['Cat'] = df['Cat'].apply(lambda x: 1 if x =='val' else 0)
    return df

def create_dataloaders(df: pd.DataFrame, size: Tuple[int, int, int], bs: int) -> DataLoaders:
    """Create 3D image dataloaders with specified size and batch size."""
    return ImageDataLoaders3D.from_df(
        df, 
        path=IMG_PATH,
        fn_col=DATA_ID,
        label_col=LABEL_ID,
        valid_col='Cat',
        item_tfms=[ResizeCrop3D(crop_by=(0, 0, 0), resize_to=size),
                  *TioTransforms(p_all=0.2)],
        batch_tfms=aug_transforms_3d(p_all=0.2),
        bs=bs, val_bs=bs
    )

def get_training_callbacks(model_name: str, log_name: str) -> list:
    """Get standard training callbacks."""
    return [
        Recorder(train_metrics=True),
        SaveModelCallback(
            monitor='valid_roc_auc_score', 
            with_opt=True,
            fname=model_name
        ),
        CSVLogger(fname=MODEL_DIR/log_name),
        GradientClip(),
        GradientAccumulation(),
        ReduceLROnPlateau(patience=5, monitor='valid_roc_auc_score', factor=50),
        EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=100)
    ]

def train_model(learn: Learner, lr: float, epochs: int) -> None:
    """Train model with flat cosine learning rate schedule."""
    learn.fit_flat_cos(epochs, lr)
    learn.unfreeze()
    learn.fit_flat_cos(epochs, lr=slice(lr*1e-3, lr*1e-1))

def main():
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    
    # Load and preprocess data
    train_df = preprocess_labels(pd.read_csv(PROCESSED_DATA/TRAIN_CSV))
    test_df = preprocess_labels(pd.read_csv(PROCESSED_DATA/TEST_CSV))
    train_df = setup_kfold(train_df)
    
    # Stage 1 training
    dls = create_dataloaders(train_df, (16, 32, 32), BATCH_SIZE1)
    learn = cnn_learner_3d(
        dls, 
        ARCH, 
        metrics=[accuracy, F1Score(), RocAucBinary()],
        opt_func=ranger,
        loss_func=LabelSmoothingCrossEntropy(),
        model_dir=MODEL_DIR,
        cbs=[]
    )
    learn.add_cbs(get_training_callbacks(MODEL_NAME_TEMPLATE, HISTORY_LOG_TEMPLATE_STAGE1))
    learn.to_fp16()
    train_model(learn, LR_VALLEY, NUM_EPOCH1)
    
    # Stage 2 training
    dls = create_dataloaders(train_df, (32, 64, 64), BATCH_SIZE2)
    learn = cnn_learner_3d(
        dls, 
        ARCH, 
        metrics=[accuracy, F1Score(), RocAucBinary()],
        opt_func=ranger,
        loss_func=LabelSmoothingCrossEntropy(),
        model_dir=MODEL_DIR,
        cbs=[]
    )
    learn.add_cbs(get_training_callbacks(FINAL_MODEL_NAME_TEMPLATE, HISTORY_LOG_TEMPLATE_STAGE2))
    learn.to_fp16()
    learn = learn.load(STAGE1_MODEL_LOAD_TEMPLATE)
    train_model(learn, LR_VALLEY, NUM_EPOCH2)

if __name__ == '__main__':
    main()
