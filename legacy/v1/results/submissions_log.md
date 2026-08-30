# Submissions Log

This file tracks every official submission made during the **QRT Asset Allocation Performance Forecasting Challenge**.

The public leaderboard is considered **only as an external validation signal**.

All model selection decisions are primarily based on:

- temporal expanding-window validation;
- local cross-validation metrics;
- scientific interpretation of the results.

The objective is **not** to maximize the number of submissions but to submit only models that demonstrate convincing local improvements.

---

## Official Submissions

| ID | Date | Model | Feature Set | Local Accuracy | Local ROC-AUC | Public Score | Decision | Comments |
|:--:|:----------:|-------------------------|--------------------------------|:-------------:|:-------------:|:------------:|--------------------|--------------------------------------------------------------|
| 001 | 2026-07-10 | Logistic Regression | RET_1 ... RET_20 | ~0.51 | ~0.51 | **0.50185** | Reference | First official submission after the feature engineering stage. |
| 002 | 2026-07-12 | Gradient Boosting | RET_1 ... RET_20 | ~0.525 | ~0.533 | **0.50957** | **Current Best** | Best public score obtained so far. Demonstrates the benefit of nonlinear boosting over linear models. |
| 003 | 2026-07-14 | XGBoost | RET_1 ... RET_20 | ~0.525 | ~0.533 | **0.50712** | Rejected | Performance below the current Gradient Boosting reference despite similar local validation metrics. |
| 004 | 2026-07-14 | XGBoost | All engineered numerical features | ~0.525 | ~0.533 | *Pending* | Under Evaluation | Evaluates whether richer feature engineering improves advanced boosting models. |

---

# Experimental Summary

The project has progressively evaluated several Machine Learning model families.

## Baselines

- Majority class
- Always positive
- Always negative
- Momentum
- Mean momentum
- Mean reversion

These baselines establish the minimum expected performance.

---

## Linear Models

### Logistic Regression

The logistic regression model provides a strong linear baseline.

It captures simple relationships between historical returns and the binary target but remains limited when nonlinear interactions are present.

---

## Tree-Based Models

### Decision Tree

A single decision tree captures nonlinear relationships but suffers from high variance.

---

### Random Forest

Random Forest improves stability through bagging and feature randomness.

Performance improves compared to a single tree but remains below the best boosting models.

---

## Classical Boosting

### AdaBoost

AdaBoost provides only a modest improvement.

The sequential correction of misclassified observations appears insufficient to extract substantially more signal from the current feature representation.

---

### Gradient Boosting

Gradient Boosting currently remains the strongest model.

Main observations:

- captures nonlinear interactions;
- improves over Logistic Regression;
- produces the best public leaderboard score obtained so far;
- remains the reference model for the project.

Current best public score:

**0.50957**

---

## Advanced Boosting

Three modern implementations were investigated.

### XGBoost

Experiments were conducted using:

- historical returns only;
- returns + momentum;
- returns + signed volumes;
- complete numerical feature set.

Despite its regularized objective function and second-order optimization, XGBoost did **not** consistently outperform the previously retained Gradient Boosting model.

---

### LightGBM

LightGBM produced performances extremely close to XGBoost.

The histogram-based algorithm and leaf-wise growth strategy did not produce a significant advantage under the current experimental conditions.

---

### CatBoost

The pipeline has been implemented.

A complete experimental study using native categorical variables remains a potential future extension.

---

# Current Scientific Conclusions

The experiments suggest that:

- nonlinear boosting models outperform linear models;
- Gradient Boosting remains the strongest model evaluated so far;
- XGBoost and LightGBM produce very similar performances;
- changing the boosting implementation alone is not sufficient to improve the leaderboard score;
- the current limitation appears to be related more to the available predictive signal than to the learning algorithm itself.

---

# Future Experiments

The next research directions include:

- understanding the official evaluation metric more precisely;
- probability calibration;
- feature importance analysis;
- SHAP explanations;
- model ensembles;
- stacking;
- ranking-based objectives;
- improved feature engineering;
- deployment with Streamlit, FastAPI and Docker.

---

Last update:

**Advanced Boosting Models (Step 09)**