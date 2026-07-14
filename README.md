# QRT Asset Allocation Performance Forecasting

> **Machine Learning for Financial Asset Allocation using Temporal Validation and Advanced Boosting Models**

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Finance](https://img.shields.io/badge/Domain-Quantitative%20Finance-darkgreen)
![Status](https://img.shields.io/badge/Status-In%20Progress-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Overview

This repository contains my complete solution to the **QRT Asset Allocation Performance Forecasting Challenge**.

The objective of the challenge is to predict whether a daily asset allocation will generate a **positive future return**, using only historical market information.

Unlike many traditional Machine Learning projects, this challenge requires respecting the temporal structure of financial data. Every experiment therefore follows a strict chronological validation protocol to avoid look-ahead bias and data leakage.

The project progressively investigates increasingly sophisticated Machine Learning models while maintaining a rigorous scientific methodology.

---

## Objectives

The project aims to answer several research questions:

- Can historical returns predict future allocation performance?
- How much information is contained in engineered financial features?
- Do nonlinear models outperform linear models?
- Can modern boosting algorithms improve classical Gradient Boosting?
- How should temporal validation be performed in quantitative finance?

---

# Project Structure

```
QRT_Asset_Allocation/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── submissions/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_validation_strategy.ipynb
│   ├── 04_baselines.ipynb
│   ├── 05_logistic_regression.ipynb
│   ├── 06_feature_engineering.ipynb
│   ├── 07_tree_models.ipynb
│   ├── 08_boosting_models.ipynb
│   └── 09_advanced_boosting_models.ipynb
│
├── src/
│
├── README.md
├── requirements.txt
└── submissions_log.md
```

---

# Scientific Methodology

The project follows a strict experimental workflow.

Every modelling stage follows exactly the same protocol.

```
Data Understanding
        ↓
Exploratory Data Analysis
        ↓
Temporal Validation Strategy
        ↓
Baselines
        ↓
Linear Models
        ↓
Feature Engineering
        ↓
Tree Models
        ↓
Boosting Models
        ↓
Advanced Boosting
        ↓
Model Comparison
        ↓
Official Submission
```

The public leaderboard is **never used to choose a model**.

Model selection relies exclusively on local temporal validation.

The leaderboard is only used as an external validation signal.

---

# Temporal Validation

Financial observations are ordered in time.

Random train/test splits would introduce severe look-ahead bias.

The project therefore uses an **Expanding Window Validation** strategy.

For every fold:

- the training period always precedes the validation period;
- validation windows contain 120 trading dates;
- the training window expands after each fold;
- no future information is used.

---

# Models Evaluated

## Baselines

- Majority Class
- Always Positive
- Always Negative
- Momentum
- Mean Momentum
- Mean Reversion

---

## Linear Models

- Logistic Regression

---

## Tree-Based Models

- Decision Tree
- Random Forest

---

## Classical Boosting

- AdaBoost
- Gradient Boosting

---

## Advanced Boosting

- XGBoost
- LightGBM
- CatBoost

---

# Feature Engineering

Several financial feature families were investigated.

### Historical Returns

```
RET_1 ... RET_20
```

### Momentum

- momentum_3
- momentum_5
- momentum_10
- momentum_20

---

### Volatility

- volatility_3
- volatility_20

---

### Volume Features

- volume_3
- volume_20
- delta_volume

---

### Risk Adjusted Features

- momentum_volatility_ratio

---

### Signed Volumes

Historical signed trading volumes.

---

# Evaluation Metrics

Each model is evaluated using:

- Accuracy
- ROC-AUC
- Log-Loss

The same metrics are computed on every temporal fold.

---

# Main Results

Several important observations emerged during the project.

### Logistic Regression

Provides a strong linear baseline.

---

### Random Forest

Captures nonlinear interactions but provides only moderate improvements.

---

### Gradient Boosting

Produces the best overall performance.

This suggests that nonlinear additive models better capture the weak financial signal.

---

### XGBoost

Despite its sophisticated optimization strategy, XGBoost did not consistently outperform the classical Gradient Boosting model under the current feature representation.

---

### LightGBM

LightGBM produced results extremely close to XGBoost.

No significant advantage was observed.

---

### CatBoost

Pipeline implemented.

Further experiments may investigate native categorical features.

---

# Best Public Leaderboard Score

Current best public score:

```
0.50957
```

Obtained with:

- Gradient Boosting
- Historical Returns (RET_1 ... RET_20)

---

# Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-Learn
- XGBoost
- LightGBM
- CatBoost
- Jupyter Notebook

---

# Installation

Clone the repository

```bash
git clone https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Launch the notebooks sequentially.

The recommended execution order is:

```
01
↓

02
↓

03
↓

04
↓

05
↓

06
↓

07
↓

08
↓

09
```

Each notebook is independent and documents its conclusions before moving to the next stage.

---

# Future Work

Several research directions remain open.

- Better understanding of the official challenge metric.
- Probability calibration.
- Model ensembles.
- Stacking.
- Ranking-based learning objectives.
- SHAP value interpretation.
- Hyperparameter optimization.
- Streamlit application.
- FastAPI deployment.
- Docker containerization.

---

# Repository Philosophy

This repository focuses on understanding **why** a model works rather than simply obtaining a higher leaderboard score.

Every modelling decision is documented, interpreted and compared under identical temporal validation conditions.

The goal is to build reproducible and scientifically justified Machine Learning experiments for quantitative finance.

---

# Author

**Phillipe BAGUEKA**
