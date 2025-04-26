# Lung-PNetV2 Image Classification Module

## Project Description
This module uses a 3D CNN model to classify lung CT images, distinguishing between IAC (Invasive Adenocarcinoma) and other types.

## Installation
```bash
pip install -r requirements.txt
```

## Configuration
Modify path and parameter settings in `config.py`:
- `NB_DIR`: Root data directory
- `LOCAL_DATA`: Processed data directory
- `MODEL_DIR`: Model save directory

## Training
```bash
python train.py
```

## Parameters
- `REGION_NAME`: Region type (roi/merge)
- `METHOD_NAME`: Processing method (DIL/ERO)
- `CLIP`: CT Window Level cropping parameters
- `N_FOLDS`: Number of K-fold cross-validation
- `FOLD`: Current fold number
- `ARCH`: Model architecture

## Output
- Training metrics are recorded in `history_*.csv`
- Best models are saved in `MODEL_DIR` directory

## Visualizations
![Sample CT Scan](docs/images/sample_ct_scan.png)
*Example lung CT scan showing IAC features*

![Model Architecture](docs/images/model_architecture.png)
*3D CNN model architecture diagram*

