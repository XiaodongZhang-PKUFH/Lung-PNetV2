# config.py
from pathlib import Path

# General settings
clip = [50, 50]  # Clipping values for training

# Path configurations
# Base data directory
NB_DIR = Path('path_to_data')
# Processed high quality data directory
LOCAL_DATA = NB_DIR/'processed'
PROCESSED_DATA = LOCAL_DATA
IMG_PATH = LOCAL_DATA
# Pre-trained model weights directory
path_model_state_dict = Path('path_to_model_weights')
# Coronal CT slices directory
DATA_PATH = Path('path_to_coronal_ct_slices')
# Raw 3D NII data directory
RAW_DATA_PATH = Path('path_to_raw_3d_nii_data')
# Default output directory for processed data
DEFAULT_OUTPUT_DIR = Path('path_to_processed_data')
# CSV files output directory
CSV_OUTPUT_DIR = Path('path_to_csv_files')
# Training dataset CSV file
TRAIN_CSV = 'CT_HQ_LQ_train_lists_resolution_enhancement.csv'
# Validation dataset CSV file
VALID_CSV = 'CT_HQ_LQ_valid_lists_resolution_enhancement.csv'
# Test dataset CSV file
TEST_CSV = 'CT_HQ_LQ_test_lists_resolution_enhancement.csv'
# Holdout test datasets directory
HOLDOUT_DATA_DIR = Path('path_to_holdout_test_datasets')

# Model training settings
N_FOLDS = 5
SEED = 41
fold = 2
dataID = 7
labelID = 38
num_epoch1 = 30
num_epoch2 = 100
beta = 0.99
wd = 1e-3
lr = 1e-3
