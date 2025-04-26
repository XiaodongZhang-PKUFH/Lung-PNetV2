# config.py
import pathlib
from torchvision.models.video import r3d_18


# General settings
CLIP = [50, 50]  # train:[0,0],[50,50],[100,100]; test: [0,0]

# Path settings
NB_DIR = pathlib.Path('path_to_data')
LOCAL_DATA = NB_DIR/'processed'
PROCESSED_DATA = LOCAL_DATA
IMG_PATH = LOCAL_DATA
TRAIN_CSV = f'pggn_train_swn_{CLIP[0]}_{CLIP[1]}.csv'
TEST_CSV = f'pggn_test_swn_0_0.csv'

# Model training settings
N_FOLDS = 5  # 5,3
SEED = 41
FOLD = 2  # 0,1,2,3,4
DATA_ID = 7
LABEL_ID = 38
NUM_EPOCH1 = 30
NUM_EPOCH2 = 100
BATCH_SIZE1 = 16
BATCH_SIZE2 = 8
LR_VALLEY = 3e-3

# Model architecture
ARCH = r3d_18  
MODEL_DIR = PROCESSED_DATA/'models3d/classification_kfold'
MODEL_NAME_TEMPLATE = 'model_resized_pggn_swn_{CLIP[0]}_{CLIP[1]}_fold_{FOLD}_image_only'
STAGE1_MODEL_LOAD_TEMPLATE = 'model_resized_pggn_swn_{CLIP[0]}_{CLIP[1]}_fold_{FOLD}_image_only'
FINAL_MODEL_NAME_TEMPLATE = 'final_model_pggn_swn_{CLIP[0]}_{CLIP[1]}_fold_{FOLD}_image_only'
HISTORY_LOG_TEMPLATE_STAGE1 = 'history_pggn_swn_{CLIP[0]}_{CLIP[1]}_fold_{FOLD}_image_only_stage1.csv'
HISTORY_LOG_TEMPLATE_STAGE2 = 'history_pggn_swn_{CLIP[0]}_{CLIP[1]}_fold_{FOLD}_image_only_stage2.csv'

# SSL model paths
PATH_MODEL_STATE_DICT = Path('path_to_ssl_model_state_dict.pth')

# Model loading template
FINAL_MODEL_LOAD_TEMPLATE = 'final_model_pggn_swn_{CLIP[0]}_{CLIP[1]}_fold_{FOLD}_image_only'
