# QRT Asset Allocation Performance Forecasting

## Overview

This repository contains my solution to the **QRT Asset Allocation Performance Forecasting Challenge**.

The objective of this project is not only to build a predictive model, but to conduct a **reproducible quantitative research workflow** for asset allocation forecasting.

Rather than comparing machine learning models in isolation, each stage of the project is driven by a scientific hypothesis, validated through a strict temporal evaluation protocol and fully documented before moving to the next step.

The repository is developed incrementally following professional software engineering practices, including modular code organization, reusable components, experiment reproducibility and version control.

---

# Research Workflow

```
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Temporal Validation Strategy
        ↓
Deterministic Baselines
        ↓
Logistic Regression (Raw Returns)
        ↓
Feature Engineering
        ↓
Regularized Logistic Regression
        ↓
Tree-Based Models
        ↓
Hyperparameter Optimization
        ↓
Model Interpretation
        ↓
Streamlit + FastAPI
        ↓
Docker Deployment
```

---

# Current Project Status

## Completed

### 01 — Data Understanding

- Dataset inspection
- Target understanding
- Feature identification
- Dataset structure analysis

---

### 02 — Exploratory Data Analysis

- Target distribution
- Temporal analysis
- Allocation analysis
- Group analysis
- Preliminary signal investigation

---

### 03 — Temporal Validation Strategy

A robust chronological validation protocol has been implemented using an **expanding-window** strategy.

Main characteristics:

- strict chronological split
- expanding training window
- fixed validation horizon
- leakage prevention
- reusable validation utilities

---

### 04 — Deterministic Baselines

Several reference strategies have been implemented to establish meaningful performance baselines:

- Majority Class
- Always Positive
- Always Negative
- Momentum
- Reversal
- Aggregated Momentum
- Aggregated Reversal

All strategies are evaluated using the same temporal validation framework.

---

### 05 — Logistic Regression on Raw Returns

Implementation of the first supervised learning model.

Pipeline:

```
SimpleImputer
        ↓
StandardScaler
        ↓
LogisticRegression
```

The evaluation framework computes:

- Accuracy
- ROC-AUC
- Log-Loss

using the same temporal validation protocol adopted for all previous experiments.

---

# Main Scientific Findings

The first Logistic Regression model uses the twenty raw historical returns (`RET_1` ... `RET_20`) as predictive features.

The obtained results show:

- performances very close to the Momentum baseline;
- ROC-AUC only slightly above random guessing;
- Log-Loss close to that of an uninformative classifier.

These observations suggest that **raw returns contain only limited predictive information in their current representation**.

Consequently, the next research hypothesis focuses on **Feature Engineering** rather than immediately increasing model complexity.

---

# Project Structure

```
.
├── data/
│   └── raw/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_validation_strategy.ipynb
│   ├── 04_baselines.ipynb
│   ├── 05_logistic_regression.ipynb
│   └── benchmark_submission.ipynb
│
├── src/
│   ├── data_loading.py
│   ├── validation.py
│   ├── baselines.py
│   ├── modeling.py
│   └── evaluation.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Software Architecture

The project follows a modular architecture where notebooks only orchestrate experiments.

Business logic is centralized inside the `src/` package.

Current modules:

| Module            | Responsibility                         |
| ----------------- | -------------------------------------- |
| `data_loading.py` | Dataset loading utilities              |
| `validation.py`   | Temporal validation strategy           |
| `baselines.py`    | Deterministic benchmark strategies     |
| `modeling.py`     | Machine Learning pipeline construction |
| `evaluation.py`   | Baseline and model evaluation          |

This separation improves:

- maintainability;
- reproducibility;
- code reuse;
- experiment consistency.

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Jupyter Notebook

---

# Repository Philosophy

This repository is intentionally developed **incrementally**.

Each major research stage follows the same workflow:

1. formulate a scientific hypothesis;
2. implement the experiment;
3. evaluate using temporal validation;
4. interpret the results;
5. document the conclusions;
6. refactor the code;
7. commit and publish the milestone.

The objective is not only to maximize predictive performance, but also to understand **why** a model succeeds or fails.

---

# Next Research Steps

- Feature Engineering
- Regularized Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost / LightGBM
- Hyperparameter Optimization
- Model Interpretation
- Model Persistence
- FastAPI
- Streamlit Dashboard
- Docker Deployment

---

# Author

**Phillipe BAGUEKA**
