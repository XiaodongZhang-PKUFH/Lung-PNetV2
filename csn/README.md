# Lung-PNetV2 CSN Python Module

## Project Description
This project contains Python scripts for processing lung CT scan data, including 2D/3D image processing and resolution enhancement.

## Requirements
- Python 3.7+
- See requirements.txt for dependencies

## Installation
1. Create conda environment:
```bash
conda create -n lungpnet python=3.8
conda activate lungpnet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure paths:
Modify path settings in config.py:
- `NB_DIR`: Base data directory
- `path_model_state_dict`: Pretrained model weights directory
- Other relevant paths

## Usage

### 1. Generate 3D Slices
```bash
python batch_create_3d_nii.py
```

### 2. Generate 2D NII Files
```bash
python batch_create_2d_slices.py
```

### 3. Chest CT Resolution Enhancement
```bash
python chest_resolution_enhancement.py
```

## Notes
1. Ensure all input data paths are properly configured
2. Training parameters can be adjusted in config.py
3. Output directory needs sufficient storage space

## Visualizations
![Project Diagram](docs/images/project_diagram.png)
