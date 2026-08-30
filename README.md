# QRT Challenge Data — Predicting Asset Allocation Return Direction

[![Python 3.11-3.13](https://img.shields.io/badge/Python-3.11--3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://docs.python.org/3/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docs.docker.com/)
[![Quantitative Finance](https://img.shields.io/badge/Quantitative%20Finance-Challenge%20Data-3B4CCA?style=flat-square)](https://challengedata.ens.fr/challenges/167)

## Overview

This is an end-to-end binary-classification project for the QRT Challenge Data
asset-allocation problem. It predicts whether an allocation's next return will
be positive. The repository covers group-aware validation, reproducible model
selection, a persisted scikit-learn pipeline, tested FastAPI inference, a
Streamlit demonstration, Docker Compose, and continuous integration.

The final model is intentionally small and understandable. The project is
closed as a portfolio case study: methodological discipline and reliable
serving take precedence over further leaderboard optimization.

## Challenge and data

The target is the sign of a future continuous return: positive values map to
class `1`; zero or negative values map to class `0`. Inputs include twenty raw
lagged returns (`RET_1` to `RET_20`), signed volumes, allocation identifiers,
and anonymized `TS` groups. The final model uses only the twenty returns.

The labelled dataset contains 527,073 rows in 2,522 `TS` groups. Raw challenge
CSVs are deliberately excluded from Git. See the
[official challenge](https://challengedata.ens.fr/challenges/167) for access
and terms. This repository is a personal, unofficial solution and is not
affiliated with QRT or Challenge Data.

## From V1 to V2

| Area | V1 | V2 (active) |
|---|---|---|
| Validation assumption | Expanding windows based on a presumed order of anonymized `TS` values | `TS` treated only as an opaque group |
| Development protocol | Four temporal-style folds | Five-fold `StratifiedGroupKFold` on development |
| Final assessment | Public leaderboard used as an external signal | Frozen 20% grouped lockbox evaluated after model freeze |
| Workflow | Notebook-led experimentation | Configurations, scripts, tests, reports, and hashes |
| Serving | Local V1 artifact | Versioned final V2 artifact |
| Reproducibility | Historical research snapshot | Notebook-free pipeline, Docker, and CI |

V1's chronological interpretation cannot be defended from the anonymized data
alone. This does not prove that V1 leaked; it means its core assumption and the
associated methodological risk were insufficiently controlled. V1 remains
read-only under `legacy/v1/` at source commit
`e64aa6064e723bad935651bd0c664ea9c044ffc0`.

## Validation strategy

All rows sharing a `TS` stay together. The frozen assignment first separates
421,654 development rows (2,028 groups) from 105,419 lockbox rows (494 groups).
Five definitive stratified group folds exist only inside development. Model and
feature decisions use development out-of-fold predictions; the lockbox is not
used for selection. The final pipeline is then refitted on all labelled rows.

```text
Labelled data
|-- Development (80%) -> five group-aware OOF folds -> model freeze
`-- Lockbox (20%)     -> one final assessment after freeze
                                      |
                                      v
                             full-train refit
```

The first lockbox computation was lost before persistence because a downstream
test CSV used `;` rather than `,`. An explicitly authorized, identical technical
rerun recorded the metrics atomically. No first-attempt metric was observed,
and no model, feature, parameter, seed, threshold, or decision changed between
attempts. See the immutable lockbox report for the full incident record.

## Final model

The persisted pipeline contains an ordered `RET_1`-to-`RET_20` selector, a
constant-zero `SimpleImputer`, and scikit-learn's
`GradientBoostingClassifier`. It uses:

| Parameter | Value |
|---|---:|
| `n_estimators` | 50 |
| `learning_rate` | 0.05 |
| `max_depth` | 2 |
| `min_samples_leaf` | 20 |
| `subsample` | 0.7 |
| `max_features` | `sqrt` |
| `random_state` | 42 |
| decision threshold | 0.5 |

No scaler is needed for this tree model. The final choice deliberately favors
simplicity, reproducibility, and a clear inference contract.

## Results

| Scope | Accuracy | ROC-AUC | Log-loss |
|---|---:|---:|---:|
| Development OOF (421,654 rows) | 0.520005 | 0.526351 | 0.692253 |
| Final lockbox (105,419 rows) | 0.520087 | 0.527912 | 0.692218 |

The lockbox confusion matrix is TN=16,931, FP=35,005, FN=15,587,
TP=37,896. Its observed positive rate is 0.507337 and predicted positive rate
is 0.691536. The historical V1 public leaderboard score was approximately
0.50957; it is a different external metric and is not directly comparable to
local ROC-AUC or accuracy.

The final artifact was refitted on all 527,073 labelled rows, so it has no
additional honest local performance estimate. The signal is weak and remains
close to chance. This is not evidence of a profitable trading strategy.

## Architecture

```text
Streamlit :8501 -> FastAPI :8000 -> scikit-learn Pipeline -> JSON prediction
```

```text
app/                    FastAPI inference service
frontend/               Streamlit API client
models/                 Versioned final Joblib and manifest
src/qrt_forecasting/    Active reusable package
configs/                Frozen experiment and final configurations
reports/                Validation, experiment, and final reports
tests/                  Unit, methodology, API, and integration tests
legacy/v1/              Read-only historical V1
```

The official benchmark notebook remains under `research/references/` for
read-only context. No notebook is required to train, evaluate, serve, or test
the final model.

## Run locally

Python 3.11 through 3.13 is supported; CI and Docker use Python 3.13.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[app,dev]"
uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```bash
streamlit run frontend/streamlit_app.py
```

Run tests with `python -m pytest -q`.

## Run with Docker

```bash
docker compose build
docker compose up -d
```

Open FastAPI at <http://localhost:8000/docs> and Streamlit at
<http://localhost:8501>. Stop both services with `docker compose down`.

## API example

`POST /predict` accepts exactly twenty fields. A minimal payload sets each to
zero:

```json
{
  "RET_1": 0.0, "RET_2": 0.0, "RET_3": 0.0, "RET_4": 0.0,
  "RET_5": 0.0, "RET_6": 0.0, "RET_7": 0.0, "RET_8": 0.0,
  "RET_9": 0.0, "RET_10": 0.0, "RET_11": 0.0, "RET_12": 0.0,
  "RET_13": 0.0, "RET_14": 0.0, "RET_15": 0.0, "RET_16": 0.0,
  "RET_17": 0.0, "RET_18": 0.0, "RET_19": 0.0, "RET_20": 0.0
}
```

Example response:

```json
{
  "model_name": "gradient_boosting_ret20_final",
  "model_version": "2.0.0",
  "positive_probability": 0.51,
  "predicted_class": 1,
  "threshold": 0.5
}
```

The API also exposes `GET /health` and `GET /model-info`.

## Reproducibility

The repository tracks frozen TOML configurations, seed 42, the grouped-fold
manifest, canonical hashes, OOF reports, an immutable lockbox incident report,
the small final model, and its manifest. Submission files remain local and are
validated against the official schema, reopened after writing, and hashed.
Tests cover data contracts, grouping invariants, pipeline parameters,
serialization, API behavior, and platform-safe text hashing.

## Limitations

- Predictive signal is weak and close to 0.5.
- Data and group labels are anonymized; no reliable chronology is inferred.
- Local validation and leaderboard behavior can differ.
- The model is educational and does not represent financial advice.
- No return, risk, or profitability guarantee is made.

## Project status

The portfolio version is closed. Advanced feature engineering, alternative
boosting engines, hyperparameter optimization, deep learning, and renewed
leaderboard competition are outside scope. Historical research remains for
traceability but is not the production pipeline.
