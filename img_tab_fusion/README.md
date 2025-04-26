# Image and Tabular Data Fusion Module

This Python module implements a fusion model combining 3D image features and tabular clinical data for lung nodule classification.

## Features
- 3D CNN for image feature extraction
- Tabular model for clinical data processing
- Fusion layer combining both modalities

## Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration
Edit `config.py` to set:
- Data paths
- Model parameters
- Training hyperparameters

## Usage
To train the model:
```bash
python train.py
```

## Model Architecture
The model combines:
- 3D CNN for image feature extraction
- Tabular model for clinical data
- Fusion layer combining both modalities

## Outputs
Training produces:
- Model checkpoints
- Training history logs
- Performance metrics

## Requirements
See `requirements.txt` for full dependency list.
