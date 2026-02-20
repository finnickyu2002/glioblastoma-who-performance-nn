# Glioblastoma WHO Performance Status Prediction (MLP)

## Overview

This project implements a neural network–based classification pipeline to predict **WHO performance status** in patients with glioblastoma (GBM) using structured clinical, treatment, and imaging availability data.

The objective is to model functional status from multi-source tabular patient data using a reproducible machine learning workflow.

---

## Problem Formulation

Given integrated structured patient-level features:

- Clinical variables (demographics, baseline measurements)
- Treatment parameters (e.g., radiotherapy information)
- Imaging availability indicators

Predict the categorical outcome:

> WHO performance status

This is formulated as a supervised multi-class classification problem.

---

## Methodology

### 1. Data Integration

- Merge clinical, treatment, and imaging datasets
- Deduplicate imaging entries per patient
- Remove samples with missing target labels

### 2. Preprocessing

- Median imputation for numerical features
- Mode imputation for categorical features
- One-hot encoding of categorical variables
- Train/test feature alignment
- Feature standardization using `StandardScaler`

### 3. Model

Multi-Layer Perceptron (MLP) classifier:

- Hidden layers: (32, 8)
- L2 regularization
- Controlled learning rate
- Stratified 80/20 train–test split

---

## Evaluation

- Accuracy
- Precision, Recall, F1-score
- Full classification report

---

## Reproducibility

The serialized model package includes:

- Trained MLP model
- Scaler
- Feature column ordering
- Imputation values
- Target metadata

Saved using `joblib` to ensure consistent inference.

---

## Project Structure

── train_who_nn.py
── who_performance_status_nn.joblib
── README.md
── LICENSE

---

## Running the Model

```bash
python train_who_nn.py
