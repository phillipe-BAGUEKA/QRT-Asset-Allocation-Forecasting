# QRT Asset Allocation Performance Forecasting

## Overview

This repository contains my solution to the **QRT Asset Allocation Performance Forecasting Challenge**.

The objective of this project is not only to build a predictive model, but to conduct a **reproducible quantitative research workflow** for asset allocation forecasting.

Rather than comparing machine learning models in isolation, each stage of the project is driven by a scientific hypothesis, evaluated through a strict temporal validation protocol and documented before moving to the next step.

The repository is developed incrementally following professional software engineering practices, including modular code organization, reusable components, experiment reproducibility and version control.

---

## Research Workflow

```text
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
Official Submission Pipeline
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

## Current Project Status

### Completed

#### 01 — Data Understanding

- Dataset inspection
- Target understanding
- Feature identification
- Dataset structure analysis

#### 02 — Exploratory Data Analysis

- Target distribution analysis
- Temporal analysis
- Allocation-level analysis
- Group-level analysis
- Preliminary signal investigation

#### 03 — Temporal Validation Strategy

A robust chronological validation protocol has been implemented using an **expanding-window** strategy.

Main characteristics:

- strict chronological split;
- expanding training window;
- fixed validation horizon;
- leakage prevention;
- reusable validation utilities.

#### 04 — Deterministic Baselines

Several reference strategies have been implemented to establish meaningful performance baselines:

- Majority Class
- Always Positive
- Always Negative
- Momentum
- Reversal
- Aggregated Momentum
- Aggregated Reversal

All strategies are evaluated using the same temporal validation framework.

#### 05 — Logistic Regression on Raw Returns

Implementation of the first supervised learning model.

Pipeline:

```text
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

#### 06 — Feature Engineering

A first complete feature engineering phase has been conducted using the logistic regression pipeline as a fixed measurement instrument.

The objective was to isolate the effect of the engineered features from the effect of model complexity.

Feature families tested:

- multi-horizon momentum;
- realized volatility;
- momentum / volatility ratio;
- signed volume.

Each feature family was introduced through an economic hypothesis, evaluated independently and compared against the logistic regression model trained on raw historical returns.

Main conclusion:

None of the tested feature families provided a robust improvement over the raw return representation under the current linear model.

This result does not imply that momentum, volatility or volume are irrelevant. It only indicates that the current representations did not add exploitable information for a logistic regression model beyond what was already contained in `RET_1` to `RET_20`.

#### 07 — First Official Submission Pipeline

A reproducible submission pipeline has been created.

The pipeline:

- loads the train and test datasets;
- creates the binary classification target;
- trains the selected final model on the full training set;
- predicts on the official test set;
- validates the expected submission format;
- exports the submission file.

The first official submission was generated using:

```text
Model: Logistic Regression
Features: RET_1 ... RET_20
```

Public leaderboard score:

```text
0.5019
```

This score is treated as a first external validation of the end-to-end pipeline, not as a feature selection criterion.

---

## Main Scientific Findings

The first logistic regression model using the twenty raw historical returns (`RET_1` ... `RET_20`) produced performance close to the deterministic momentum baseline.

The first feature engineering phase tested several economically motivated representations:

- momentum as a persistence signal;
- volatility as a proxy for local noise;
- momentum adjusted by volatility;
- signed volume as a contextual market activity signal.

However, none of these representations produced a stable improvement across temporal folds.

The current interpretation is that simple linear transformations of returns and signed volume are not sufficient to extract a stronger signal with a logistic regression model.

The next research question is therefore:

> Can non-linear models exploit interactions or threshold effects that the current linear model cannot capture?

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── X_train.csv
│   │   ├── y_train.csv
│   │   ├── X_test.csv
│   │   └── sample_submission.csv
│   │
│   └── submissions/
│       └── submission_v001.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_validation_strategy.ipynb
│   ├── 04_baselines.ipynb
│   ├── 05_logistic_regression.ipynb
│   ├── 06_feature_engineering.ipynb
│   └── benchmark_submission.ipynb
│
├── src/
│   ├── baselines.py
│   ├── data_loading.py
│   ├── evaluation.py
│   ├── features.py
│   ├── modeling.py
│   ├── submission.py
│   ├── target.py
│   └── validation.py
│
├── submissions_log.md
├── README.md
├── requirements.txt
└── .gitignore
```

Note: raw data files and generated submission files are ignored by Git.

---

## Software Architecture

The project follows a modular architecture where notebooks only orchestrate experiments.

Reusable logic is centralized inside the `src/` package.

| Module | Responsibility |
|---|---|
| `data_loading.py` | Dataset loading utilities |
| `validation.py` | Temporal validation strategy |
| `baselines.py` | Deterministic benchmark strategies |
| `modeling.py` | Machine Learning pipeline construction |
| `evaluation.py` | Baseline and model evaluation |
| `features.py` | Feature engineering utilities |
| `target.py` | Binary target creation |
| `submission.py` | Final training and submission generation |

This separation improves:

- maintainability;
- reproducibility;
- code reuse;
- experiment consistency.

---

## Methodology

Each major research stage follows the same workflow:

1. formulate a scientific hypothesis;
2. implement the experiment;
3. evaluate using temporal validation;
4. compare against relevant baselines;
5. interpret the results;
6. decide whether to keep, reject or reformulate the idea;
7. refactor reusable logic into `src/`;
8. document the conclusions;
9. commit and publish the milestone.

The objective is not only to maximize predictive performance, but also to understand **why** a model succeeds or fails.

---

## Leaderboard Policy

The public leaderboard is not used as the primary model selection tool.

Model and feature decisions are made primarily through local temporal validation.

Official submissions are used only as external checks to verify whether local conclusions are broadly consistent with out-of-sample leaderboard behavior.

This is intended to reduce the risk of leaderboard overfitting.

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Jupyter Notebook

---

## Next Research Steps

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

## Author

**Phillipe BAGUEKA**
