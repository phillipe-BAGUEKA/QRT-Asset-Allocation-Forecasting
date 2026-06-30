# QRT Asset Allocation Performance Forecasting

## Overview

This repository contains my solution to the **QRT Asset Allocation Performance Forecasting Challenge**.

The objective of this project is to build a complete machine learning pipeline capable of predicting whether the future performance of an asset allocation is positive or negative.

The project is developed progressively following good machine learning and software engineering practices. Each major stage is implemented, documented, reviewed and versioned before moving to the next step.

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
│   └── benchmark_submission.ipynb
│
├── src/
│   ├── data_loading.py
│   └── validation.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Project Progress

### Completed

- Data understanding
- Exploratory Data Analysis
- Temporal validation strategy
  - Chronological holdout
  - Expanding-window validation
  - Temporal consistency checks
  - Leakage prevention

### Upcoming Steps

- Feature engineering
- Baseline models
- Model comparison
- Hyperparameter optimization
- Model interpretation
- Streamlit application
- Docker
- Deployment

---

## Temporal Validation

A dedicated temporal validation protocol has been implemented to reproduce the chronological structure of the official challenge.

The validation pipeline includes:

- chronological train/validation split;
- expanding-window cross-validation;
- temporal leakage detection;
- automatic consistency checks.

This protocol will be reused throughout the project to ensure a fair and reproducible evaluation of every model.

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Development Process

The repository is developed incrementally.

Each major milestone follows the same workflow:

1. implementation;
2. documentation;
3. code review;
4. Git commit;
5. GitHub publication.

This approach keeps the repository organized while maintaining a clear development history.

---

## Author

**Phillipe BAGUEKA**