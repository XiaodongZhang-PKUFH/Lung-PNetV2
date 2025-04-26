import warnings
warnings.filterwarnings('ignore')

from typing import Tuple
import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import StratifiedKFold
from fastai.tabular.all import *
from fastai.callback.all import *
from faimed3d.all import *
from config import *
from model_components import AdaptiveHybridLoss, create_head_no_pool, MelModel

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataframe by adding Label2 column."""
    df['Label2'] = df['Label'].apply(lambda x: 'Other' if x == 1 else 'IAC')
    return df

def create_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Create stratified KFold splits."""
    df['fold'] = -1
    strat_kfold = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
    for i, (_, valid_index) in enumerate(strat_kfold.split(df.PatientID.values, df['Label2'].values)):
        df.iloc[valid_index, -1] = i
    df['fold'] = df['fold'].astype('int')
    return df

def create_dataloaders(train_df: pd.DataFrame, fold: int) -> Tuple[DataLoaders, TabularPandas]:
    """Create tabular and image dataloaders for given fold."""
    # Set validation fold
    train_df['Cat'] = train_df['fold'].apply(lambda x: 'val' if x == fold else 'train')
    train_df['Cat'] = train_df['Cat'].apply(lambda x: 1 if x == 'val' else 0)
    
    # Tabular dataloader
    to = TabularPandas(train_df, [Categorify, FillMissing, Normalize], 
                      CAT_NAMES, CONT_NAMES, y_names='Label2', 
                      y_block=CategoryBlock(),
                      splits=(train_df[train_df['Cat']==0].index, 
                             train_df[train_df['Cat']==1].index))
    tab_dl = to.dataloaders(bs=BS)
    
    # Image dataloader
    vis_dl = ImageDataLoaders3D.from_df(
        train_df, path=IMG_PATH, fn_col=7, label_col=38, valid_col='Cat',
        item_tfms=[ResizeCrop3D(crop_by=(0,0,0), resize_to=(32,64,64)), *TioTransforms(p_all=0.2)],
        batch_tfms=aug_transforms_3d(p_all=0.2), bs=BS, val_bs=BS)
    
    # Mixed dataloader
    return DataLoaders(
        MixedDL(tab_dl[0], vis_dl[0]),
        MixedDL(tab_dl[1], vis_dl[1])
    ), to

def train_fold(learn: Learner, fold: int) -> None:
    """Train model for one fold."""
    learn.add_cbs([
        Recorder(train_metrics=True),
        SaveModelCallback(
            monitor='valid_roc_auc_score', 
            with_opt=True,
            fname=MODEL_NAME_TEMPLATE.format(
                region=region_name,
                method=method_name,
                arch=arch.__name__,
                clip1=clip[0],
                clip2=clip[1],
                fold=fold)),
        CSVLogger(fname=MODEL_DIR/HISTORY_LOG_TEMPLATE.format(
            region=region_name,
            method=method_name,
            arch=arch.__name__,
            clip1=clip[0],
            clip2=clip[1],
            fold=fold))
    ])
    learn.to_fp16()
    learn.freeze_to(-5)
    learn.fit_flat_cos(NUM_EPOCH, INIT_LR)

def main():
    # Load and preprocess data
    train_df = preprocess_data(pd.read_csv(PROCESSED_DATA/TRAIN_CSV_TEMPLATE.format(
        method=method_name, region=region_name, clip1=clip[0], clip2=clip[1])))
    test_df = preprocess_data(pd.read_csv(PROCESSED_DATA/TEST_CSV_TEMPLATE.format(
        method=method_name, region=region_name)))
    train_df = create_folds(train_df)
    
    # Train each fold
    for fold in range(N_FOLDS):
        print(f"Training fold {fold}")
        dls, _ = create_dataloaders(train_df, fold)
        dls.to('cuda')
        
        # Initialize models
        vis_model = create_cnn_model_3d(arch=arch, n_out=2, n_in=dls.train.n_inp)
        tab_model = TabularModel([(3,3)], len(CONT_NAMES), out_sz=2, layers=[128], ps=0.2)
        
        # Load pretrained weights
        vis_model.load_state_dict(torch.load(MODEL_DIR/VIS_MODEL_LOAD_TEMPLATE.format(
            region_name=region_name, method_name=method_name, arch=arch, 
            clip=clip, fold=fold))['model'])
        tab_model.load_state_dict(torch.load(MODEL_DIR/TAB_MODEL_LOAD_TEMPLATE.format(
            fold=fold))['model'])
        
        # Create integrated model
        integrate_model = MelModel(
            tab_model.layers[0], 
            nn.Sequential(*list(vis_model.children())[:-4]), 
            create_head_no_pool(128+512, 2, lin_ftrs=[32])
        )
        
        # Create learner
        learn = Learner(
            dls, integrate_model,
            metrics=[accuracy, RocAucBinary()],
            opt_func=partial(RAdam, wd=1e-4, beta=0.99),
            loss_func=AdaptiveHybridLoss(num_classes=2, focal_gamma=2.0, 
                                       dice_smooth=1e-6, label_smooth=0.1, 
                                       auto_balance=True),
            model_dir=MODEL_DIR
        )
        
        # Train and save
        train_fold(learn, fold)
        torch.save({
            'model': integrate_model.state_dict(),
            'optimizer': learn.opt.state_dict()
        }, MODEL_DIR/MODEL_NAME_TEMPLATE.format(
            region=region_name, method=method_name, arch=arch.__name__,
            clip1=clip[0], clip2=clip[1], fold=fold))

if __name__ == '__main__':
    main()
