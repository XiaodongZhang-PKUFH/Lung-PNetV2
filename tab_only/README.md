# Lung-PNetV2 Tabular Features Module

## Overview
This module implements tabular feature-based classification for lung nodule analysis using fastai and scikit-learn. 

## Features
- Stratified k-fold cross validation
- Tabular model training with fastai
- ROC curve and confusion matrix evaluation
- Model saving and loading

## Requirements
See requirements.txt for package dependencies.

## Configuration
Edit config.py to set:
- Data paths
- Training parameters (epochs, batch size)
- Feature columns to use

## Usage
1. Configure paths and parameters in config.py
2. Install requirements: `pip install -r requirements.txt`
3. Run training script
