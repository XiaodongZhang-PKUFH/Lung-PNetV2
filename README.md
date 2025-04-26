# Lung-PNetV2


![Project Architecture](./docs/images/architecture.png)

## Project Overview
Lung-PNetv2 is a cross-modality fusion deep learning framework, for differentiating IAC from pre-invasive lesions in pGGNs, comprising four integrated computational components with optimized multi-modal feature learning:

### 1. Cross-Scanner Normalization (CSN)
![CSN Workflow](./docs/images/csn_workflow.png)
- Architecture: Combines 2D ResNet-34 encoder with U-Net-style decoder
- Function: Implements feature-aware intensity standardization through hybrid loss function (perceptual similarity + adversarial training)
- Output: Converts heterogeneous CT scans into normalized latent representations
- Main scripts: batch_create_2d_slices.py, batch_create_3d_nii.py, chest_resolution_enhancement.py

### 2. Image Only Classification
![Image Classification](./docs/images/image_classification.png)
- Architecture: 3D ResNet-18 with depthwise separable convolutions
- Function: Extracts multi-scale spatial features from volumetric non-expanded bounding boxes
- Preserves contextual information around pGGN lesions
- Main scripts: train.py, volumetric_feature_extractor.py

### 3. Tabular Only Classification
![Tabular Data](./docs/images/tabular_data.png)
- Architecture: Adaptive embedding network with feature masking
- Function: Learns dense embeddings for clinical variables and nodule characteristics
- Enhances representation robustness through reconstruction mechanisms
- Main scripts: train.py, clinical_embedding.py

### 4. Image-Tabular Fusion
![Fusion Model](./docs/images/fusion_model.png)
- Architecture: Dual-stream fusion with concatenated projection layers
- Function: Synchronizes frozen representations from both modalities
- Uses MLP with dropout (p=0.5) for clinical outcome prediction
- Maintains integrity of domain-specific features from pre-training
- Main scripts: train.py, cross_modal_fusion.py

### 1. Cross-Scanner Normalization (CSN)
![CSN Workflow](./docs/images/csn_workflow.png)
- Function: Processes lung CT scan data, including 2D/3D image processing and resolution enhancement
- Main scripts: batch_create_2d_slices.py, batch_create_3d_nii.py, chest_resolution_enhancement.py

### 2. Image Only Classification
![Image Classification](./docs/images/image_classification.png)
- Function: 3D CNN-based pulmonary nodule image classification
- Main scripts: train.py, volumetric_feature_extractor.py

### 3. Tabular Only Classification
![Tabular Data](./docs/images/tabular_data.png)
- Function: Clinical tabular data classification
- Main scripts: train.py, clinical_embedding.py

### 4. Image-Tabular Fusion
![Fusion Model](./docs/images/fusion_model.png)
- Function: Multimodal classification combining imaging and clinical data
- Main scripts: train.py, cross_modal_fusion.py

## Installation Guide

### General Requirements
- Python 3.7+
- Module-specific requirements in respective requirements.txt files

### Environment Setup
```bash
conda create -n lungpnet python=3.8
conda activate lungpnet
```

## Configuration
Modify path and parameter settings in each module's config.py, including:
- `NB_DIR`: Base data directory
- `LOCAL_DATA`: Processed data directory
- `MODEL_DIR`: Model save directory

## Parameter Description
- `REGION_NAME`: Region type (roi/merge)
- `METHOD_NAME`: Processing method (DIL/ERO)
- `CLIP`: Image cropping parameters
- `N_FOLDS`: K-fold cross-validation count
- `FOLD`: Current fold number
- `ARCH`: Model architecture

## Module Usage Instructions

### CSN Module
```bash
python batch_create_2d_slices.py
python batch_create_3d_nii.py
python chest_resolution_enhancement.py
```

### Image Classification Module
```bash
python train.py
```

### Tabular Data Module
```bash
python train.py
```

### Multimodal Fusion Module
```bash
python train.py
```

## Notes
1. Ensure all input data paths are correctly configured
2. Training parameters can be adjusted in each module's config.py
3. Output directory requires sufficient storage space
