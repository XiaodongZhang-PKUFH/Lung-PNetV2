import os
from pathlib import Path

CLIP = [50, 50]  # train:[50,50] test: [0,0]
ARCH = r3d_18

# General settings
TRAIN_CSV_TEMPLATE = 'pggn_train_swn_{clip1}_{clip2}.csv'
TEST_CSV_TEMPLATE = 'pggn_test_swn_0_0.csv'

# Model loading templates
VIS_MODEL_LOAD_TEMPLATE = 'pggn_swn_{clip[0]}_{clip[1]}_fold_{fold}_image_only.pth'
TAB_MODEL_LOAD_TEMPLATE = 'pggn_fold_{fold}_ct_feats_tab_only.pth'

# CSV path templates
TRAIN_CSV_TEMPLATE = 'pggn_train_swn_{clip1}_{clip2}_hq.csv'
TEST_CSV_TEMPLATE = 'pggn_test_swn_0_0_hq.csv'

# Path settings
NB_DIR = Path('path_to_data')
LOCAL_DATA = NB_DIR/'processed'
PROCESSED_DATA = LOCAL_DATA
IMG_PATH = LOCAL_DATA
MODEL_DIR = PROCESSED_DATA/'models3d/classification_kfold'
