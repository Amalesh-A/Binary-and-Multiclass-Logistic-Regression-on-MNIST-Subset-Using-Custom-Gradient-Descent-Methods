# Binary and Multiclass Logistic Regression on MNIST Subset Using Custom Gradient Descent Methods

## Overview

This project explores binary and multiclass classification on a subset of the MNIST handwritten digits dataset, focusing on digits **0, 1, and 2**. The core objective is to implement logistic regression models from scratch and train them using various gradient descent strategies.

## Features

- **Handcrafted Feature Extraction**:  
  Each 16×16 grayscale image is transformed into a feature vector using:
  - **Symmetry**: Measures the horizontal symmetry of the digit.
  - **Intensity**: Measures the average pixel intensity.
  - **Bias Term**: Constant feature for intercept handling.

- **Binary Classification**:  
  - Distinguishes between digits **1 and 2** using **sigmoid-based logistic regression**.
  - Labels are mapped to `+1` and `-1`.

- **Multiclass Classification**:  
  - Classifies digits **0, 1, and 2** using **softmax-based logistic regression**.
  - Labels are converted to one-hot encoded vectors.

- **Optimization Methods**:  
  Custom implementations of:
  - **Batch Gradient Descent (BGD)**
  - **Stochastic Gradient Descent (SGD)**
  - **Mini-Batch Gradient Descent (MiniBGD)**

- **Visualizations**:
  - 2D scatter plots of features (symmetry vs intensity).
  - Decision boundaries for both binary and multiclass classifiers.

## Files

- `main.py` – Driver script for training and evaluation.
- `DataReader.py` – Loads data and extracts features.
- `LogisticRegression.py` – Implements binary logistic regression.
- `LRM.py` – Implements multiclass logistic regression (softmax).
- `train_features.png` – Feature distribution plot.
- `train_result_sigmoid.png` – Decision boundary for binary classification.
- `train_result_softmax.png` – Decision boundaries for multiclass classification.

## Dataset

The dataset is a modified subset of MNIST and provided as `.npz` files with:
- `x`: 256-dimensional grayscale vectors (16×16 images).
- `y`: Labels (0, 1, 2).

## How to Run

Make sure you have Python and `matplotlib` installed. To execute the training and generate plots:

```bash
python main.py
