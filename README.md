# QRT Asset Allocation Performance Forecasting

## Overview

This repository contains my solution to the **QRT Asset Allocation Performance Forecasting Challenge**.

The objective of this project is to build a complete and reproducible machine learning pipeline capable of predicting whether the future performance of an asset allocation is positive or negative.

The project follows a professional and incremental workflow where each stage is fully implemented, documented, reviewed and versioned before moving to the next one.

---

## Project Structure

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
│   └── benchmark_submission.ipynb
│
├── src/
│   ├── data_loading.py
│   ├── validation.py
│   ├── baselines.py
│   └── evaluation.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Completed Stages

### 1. Data Understanding

- Dataset inspection
- Target understanding
- Dataset structure analysis

### 2. Exploratory Data Analysis

- Distribution analysis
- Allocation analysis
- Group analysis
- Preliminary feature investigation

### 3. Temporal Validation Strategy

A complete validation protocol was implemented using:

- chronological holdout;
- expanding-window cross-validation;
- temporal consistency checks;
- leakage prevention.

### 4. Baselines and First Signals

Several reference strategies were implemented and evaluated:

- Majority class baseline;
- Always predict class 0;
- Always predict class 1;
- Momentum baseline;
- Aggregated Momentum baseline;
- Reversal baseline;
- Aggregated Reversal baseline.

All baselines are evaluated using the same temporal validation protocol.

---

## Upcoming Stages

- Feature engineering
- Logistic Regression
- Tree-based models
- Model comparison
- Hyperparameter optimization
- Model interpretation
- Streamlit application
- Docker
- Deployment

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## Development Philosophy

This repository is developed incrementally.

Each major milestone follows the same workflow:

1. implementation;
2. validation;
3. interpretation;
4. documentation;
5. Git commit;
6. GitHub publication.

The objective is not only to obtain good predictive performance, but also to build a robust, interpretable and reproducible machine learning pipeline.

---

## Author

**Phillipe BAGUEKA**