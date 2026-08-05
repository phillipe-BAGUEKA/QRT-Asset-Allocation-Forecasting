# QRT Asset Allocation Performance Forecasting

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Modeling status](https://img.shields.io/badge/Modeling_pipeline-complete-brightgreen)
![Serving status](https://img.shields.io/badge/MLOps_serving-in_progress-orange)

A reproducible modeling pipeline for the QRT challenge **"Trust or Short? Predicting the Performance of Daily Asset Allocations"**.

The modeling phase is complete: the repository covers leakage-safe validation, feature and model comparisons, train-validation diagnostics, Optuna and MLflow infrastructure, and serialization of a frozen reference model. The MLOps serving layer remains to be built. This is an educational and portfolio project, not trading advice.

## Overview

The task is binary classification: predict whether the future performance of an asset allocation will be positive (`class = 1`) or negative (`class = 0`). Observations are indexed by date and allocation, so model selection uses chronological expanding-window folds instead of random splits.

The experiments consistently show a weak financial signal. Predicted probabilities tend to remain close to `0.5`, ranking is only mildly informative, and performance varies across market periods. The project therefore emphasizes validation discipline, diagnostic transparency, and reproducibility rather than headline performance.

## Dataset

| Split | Rows | Dates | Period |
|---|---:|---:|---|
| Labeled training data | 527,073 | 2,522 | `DATE_0001` to `DATE_2522` |
| Unlabeled test data | 31,870 | 120 | `DATE_2523` to `DATE_2642` |

The binary target is derived from the provided future-performance target. Its positive-class rate in the training data is approximately **50.72%**.

The final feature set contains `RET_1` through `RET_20` in exact ascending order. Raw challenge CSV files are not distributed through this repository and must be placed locally under `data/raw/`.

## Leakage-safe temporal validation

Random splitting would allow later market observations to influence validation of models trained on earlier observations. All model comparisons therefore use the same four expanding-window folds:

| Fold | Training period | Validation period |
|---:|---|---|
| 1 | `DATE_0001`-`DATE_2042` | `DATE_2043`-`DATE_2162` |
| 2 | `DATE_0001`-`DATE_2162` | `DATE_2163`-`DATE_2282` |
| 3 | `DATE_0001`-`DATE_2282` | `DATE_2283`-`DATE_2402` |
| 4 | `DATE_0001`-`DATE_2402` | `DATE_2403`-`DATE_2522` |

Each validation window contains exactly 120 dates. Training always precedes validation, the training window expands between folds, and validation periods do not overlap.

## Modeling workflow

```text
Data understanding
  -> exploratory data analysis
  -> temporal validation
  -> baselines and logistic regression
  -> feature engineering
  -> tree and boosting models
  -> advanced boosting
  -> train-validation diagnostics
  -> cross-sectional signal experiments
  -> Optuna infrastructure
  -> MLflow tracking
  -> frozen final model
  -> future MLOps serving layer
```

The notebooks preserve the exploratory reasoning, while reusable implementation and validation logic lives in `src/`. Reproducible command-line entry points live in `scripts/`.

## Feature engineering

The following feature families were investigated:

- raw historical returns, `RET_1` to `RET_20`;
- momentum over multiple horizons and long-short momentum deltas;
- rolling volatility and volatility deltas;
- momentum-to-volatility ratios;
- volume aggregations and volume deltas;
- the intra-date cross-sectional percentile of `RET_1`.

Derived temporal features and the cross-sectional percentile produced mixed fold-level results and no sufficiently robust improvement to replace the raw return set. The final model consequently uses only `RET_1` to `RET_20`.

## Models evaluated

Models evaluated through the temporal workflow include:

- majority, constant-class, momentum, and reversal baselines;
- logistic regression;
- decision tree and random forest;
- AdaBoost and scikit-learn Gradient Boosting;
- XGBoost and LightGBM;
- CatBoost experiments and pipeline support.

The repository also contains experimental PyTorch infrastructure for an MLP, a one-dimensional CNN, and an LSTM. These neural networks are not presented as validated or selected models.

## Metrics and diagnostics

Three complementary validation metrics are used:

- **Accuracy** measures classification performance at the fixed `0.5` threshold.
- **ROC-AUC** measures ranking quality independently of that threshold.
- **Log-loss** measures the quality and confidence of predicted probabilities.

Fold diagnostics report training and validation metrics, their gaps, fold sizes, model fit time, and cumulative `predict` plus `predict_proba` time for both training and validation data. This makes temporal stability and potential overfitting visible instead of relying on a single aggregate score.

## Main results and signal diagnosis

Gradient Boosting is retained as the stable reference and produced the best confirmed public score in the project. This does not mean that it dominates every local metric: advanced boosting models produced very similar results, and XGBoost was marginally stronger on some mean local ranking and probabilistic metrics.

The reference Gradient Boosting smoke trial achieved a mean training ROC-AUC of `0.53411` and a mean validation ROC-AUC of `0.52936`, a gap of approximately `0.00475`. This does not indicate strong train-fold overfitting for that configuration. However:

- performance remains close to random classification;
- probabilities are concentrated near `0.5`;
- fold results vary across periods;
- ranking is slightly informative but weak;
- repeated experimentation can still overfit the validation protocol.

### Confirmed public leaderboard results

| Model / submission | Public score |
|---|---:|
| Logistic Regression | approximately `0.50185` |
| Gradient Boosting | approximately `0.50957` |
| XGBoost, return features only | approximately `0.50712` |

These public scores are challenge scores, not local ROC-AUC or accuracy values. Decisions are guided primarily by expanding-window validation; the public leaderboard is a documented external check, not a reliable final validation set. No claim is made about private-leaderboard generalization.

## Optuna experimentation

The Optuna layer provides a reproducible `Study` composed of independent `Trial` evaluations. It uses:

- `TPESampler` with a configurable seed;
- an explicit `NopPruner`, so no pruning is active;
- `direction="maximize"`;
- mean validation ROC-AUC across the four temporal folds as the objective.

The Gradient Boosting search space is:

| Parameter | Domain |
|---|---|
| `learning_rate` | `0.01` to `0.20`, logarithmic |
| `n_estimators` | `50` to `300`, step `25` |
| `max_depth` | `1` to `4` |
| `min_samples_leaf` | `20` to `500`, logarithmic |
| `subsample` | `0.6` to `1.0`, step `0.1` |
| `max_features` | `"sqrt"`, `"log2"`, or `None` |

The infrastructure is complete and one real reference trial has been executed. No exhaustive 25- or 50-trial search was run: the limited computational budget and weak signal did not justify the cost or additional validation-selection risk.

## MLflow tracking

Experiment tracking is local and offline:

- a SQLite backend stores experiment and run metadata;
- a local artifact store contains JSON summaries;
- each optimization session creates one parent run;
- each Optuna trial creates one child run;
- parameters, metrics, tags, failure states, and trial summaries are recorded.

The reproducible runner is `scripts/run_gradient_boosting_mlflow_smoke_test.py`. Its validated one-trial result is:

| Diagnostic | Value |
|---|---:|
| Trial state | `COMPLETE` |
| Mean validation ROC-AUC | `0.52936` |
| Fold ROC-AUC | `0.51983`, `0.54985`, `0.52354`, `0.52423` |
| Standard deviation | `0.01195` |
| Worst-fold ROC-AUC | `0.51983` |
| Mean training ROC-AUC | `0.53411` |
| Mean train-validation ROC-AUC gap | `0.00475` |
| Mean validation log-loss | `0.69185` |
| Cumulative fit time | approximately `80.13` seconds |

Run the controlled smoke test from the repository root:

```powershell
python scripts/run_gradient_boosting_mlflow_smoke_test.py
```

Then open the local tracking interface with:

```powershell
.\.venv\Scripts\mlflow.exe server `
  --backend-store-uri "sqlite:///mlflow_data/tracking/mlflow.db" `
  --port 5000
```

The runner uses one trial, no parallelism, no pruning, no model registry, and no network service.

## Final model v1

The frozen reference model is named **`gradient_boosting_ret20_v1`**.

Pipeline:

1. `SimpleImputer(strategy="constant", fill_value=0.0)`
2. `GradientBoostingClassifier`

Features: `RET_1` to `RET_20`, in exact ascending order.

| Parameter | Value |
|---|---:|
| `learning_rate` | `0.05` |
| `n_estimators` | `50` |
| `max_depth` | `2` |
| `min_samples_leaf` | `20` |
| `subsample` | `0.7` |
| `max_features` | `"sqrt"` |
| `random_state` | `42` |

The validated final training run used all 527,073 labeled rows across 2,522 dates (`DATE_0001` to `DATE_2522`), with a positive-class rate of approximately `0.50718`. Fitting took approximately `23.76` seconds, and a joblib serialization round-trip was verified.

Local artifacts:

- `models/gradient_boosting_ret20_v1.joblib`
- `models/gradient_boosting_ret20_v1.metadata.json`

SHA-256 of the joblib artifact produced in the validated environment:

```text
eb192084e7cbc3e40566ba4da34fe08ed748b1266a16dfcd2e0a231ca1cba71a
```

This hash identifies that specific local artifact; binary output is not guaranteed to be byte-for-byte identical across different library or platform environments.

Regenerate the model locally with:

```powershell
python scripts/train_final_gradient_boosting.py
```

The training runner validates the input schema, Git metadata, feature order, predictions, serialization round-trip, and artifact metadata. It does not load test data or create a submission.

## Repository structure

```text
QRT-Asset-Allocation-Forecasting/
|-- data/                         # Local raw data and generated submissions; ignored
|-- mlflow_data/                  # Local SQLite tracking and artifacts; ignored
|-- models/                       # Local serialized models and metadata; ignored
|-- notebooks/
|   |-- 01_data_understanding.ipynb
|   |-- 02_eda.ipynb
|   |-- 03_validation_strategy.ipynb
|   |-- 04_baselines.ipynb
|   |-- 05_logistic_regression.ipynb
|   |-- 06_feature_engineering.ipynb
|   |-- 07_tree_models.ipynb
|   |-- 08_boosting_models.ipynb
|   |-- 09_advanced_boosting_models.ipynb
|   |-- 10_models_analysis.ipynb
|   |-- 11_signal_exploitation_cross_sectional_features.ipynb
|   `-- benchmark_submission.ipynb
|-- scripts/
|   |-- run_gradient_boosting_mlflow_smoke_test.py
|   `-- train_final_gradient_boosting.py
|-- src/
|   |-- data_loading.py, target.py
|   |-- features.py, cross_sectional_features.py
|   |-- validation.py, evaluation.py
|   |-- modeling.py, tree_models.py, boosting_models.py
|   |-- advanced_boosting.py, neural_networks.py
|   |-- optimization.py, mlflow_tracking.py
|   `-- baselines.py, submission.py
|-- tests/
|   |-- test_cross_sectional_features.py
|   |-- test_evaluation.py
|   |-- test_optimization.py
|   |-- test_mlflow_tracking.py
|   `-- test_final_gradient_boosting_training.py
|-- pytest.ini
|-- requirements.txt
|-- submissions_log.md
`-- README.md
```

## Installation

The project has been validated with Python 3.13.

```powershell
git clone https://github.com/phillipe-BAGUEKA/QRT-Asset-Allocation-Forecasting.git
cd QRT-Asset-Allocation-Forecasting

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include scikit-learn, XGBoost, LightGBM, CatBoost, PyTorch, Optuna `4.9.0`, MLflow `3.14.0`, pytest, and joblib `1.5.3`.

## Reproducing the experiments

1. Obtain the challenge data through its authorized distribution channel.
2. Place `X_train.csv`, `y_train.csv`, `X_test.csv`, and `sample_submission.csv` in `data/raw/`.
3. Follow notebooks 01 through 11 for the documented analysis path.
4. Run the Optuna/MLflow smoke test when local tracking evidence is required.
5. Regenerate the frozen final model with the training script.

Both command-line runners require execution from a clean Git worktree so their metadata describes a reproducible repository state.

## Local inference application

The project now includes a FastAPI inference service and a multipage Streamlit interface that consumes the API over HTTP. See [Running the QRT FastAPI and Streamlit Applications](docs/run_api_and_streamlit.md) for the two-terminal PowerShell setup, local URLs, and troubleshooting guidance.

## Tests and quality checks

Run the complete test suite and dependency consistency check from the activated environment:

```powershell
python -m pytest -q
python -m pip check
```

The validated state at commit `c6acba2` contains **75 passing tests** covering cross-sectional features, evaluation diagnostics, Optuna validation, MLflow tracking, and final-model training utilities.

## Local artifacts and version control

The following are intentionally ignored by Git:

- raw challenge data;
- `models/` and serialized `*.joblib` files;
- `mlflow_data/`, `mlruns/`, and `mlartifacts/`;
- locally generated submissions.

A fresh clone therefore does not include the trained model or MLflow database. After making the raw data available locally, regenerate the final model with `python scripts/train_final_gradient_boosting.py` and run the smoke-test runner if local MLflow records are needed.

## Limitations

- The predictive signal is weak and performance remains close to random classification.
- Probability estimates occupy a narrow range around `0.5`.
- Results are temporally unstable across validation periods.
- The hyperparameter search infrastructure was validated, but the search itself was intentionally incomplete.
- Advanced boosting and repeated fold evaluation have meaningful computational cost.
- Repeated experimentation creates a risk of overfitting the fixed validation protocol.
- Public leaderboard observations do not guarantee private-leaderboard performance.
- Local model and tracking artifacts are not distributed through GitHub.
- This repository is an educational and portfolio project, not financial or trading advice.

## Product roadmap

Completed modeling work:

- [x] Data understanding and exploratory analysis
- [x] Leakage-safe temporal validation
- [x] Baselines and feature engineering
- [x] Classical and advanced boosting experiments
- [x] Train-validation diagnostics
- [x] Optuna infrastructure
- [x] Local MLflow tracking
- [x] Final model serialization and metadata validation

Remaining product work:

- [ ] FastAPI inference service
- [ ] API tests and error handling
- [ ] Docker image
- [ ] Streamlit interface
- [ ] Docker Compose orchestration
- [ ] Serving documentation and deployment
