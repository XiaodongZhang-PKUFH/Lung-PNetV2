import os
from pathlib import Path

# Data configuration
NB_DIR = Path('path_to_data')
LOCAL_DATA = NB_DIR/'processed'
PROCESSED_DATA = LOCAL_DATA
IMG_PATH = LOCAL_DATA
TRAIN_CSV = 'pggn_train_swn_50_50.csv'
TEST_CSV = 'pggn_test_swn_0_0.csv'

# Model configuration
N_FOLDS = 5
SEED = 41
NUM_EPOCH = 100
BS = 16
INIT_LR = 3e-3

# Feature configuration
CAT_NAMES = ['Gender']
CONT_NAMES = [
    'Age', 'XMaxDiameter(mm)', 'Volume(ml)',
    'MaxCT', 'MinCT', 'MeanCT', 'VarCT',
    'XAreaMax(mm2)', 'DiameterMax(mm)', 'DiameterMin(mm)',
    'RatioDiameter'
]

# Output configuration
MODEL_DIR = PROCESSED_DATA/'models3d/classification_kfold'
MODEL_NAME_TEMPLATE = 'final_model_pggn_fold_{fold}_ct_feats_tab_only'
MODEL_LOAD_TEMPLATE = 'final_model_pggn_fold_{fold}_ct_feats_tab_only'
HISTORY_LOG_TEMPLATE = 'history_pggn_fold_{fold}_ct_feats_tab_only.csv'
