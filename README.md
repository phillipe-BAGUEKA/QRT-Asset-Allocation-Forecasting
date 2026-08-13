# QRT Asset Allocation Forecasting

This repository predicts whether the future performance of an asset allocation
is positive or negative. V2 is now the active research path and treats `TS` as
an opaque group identifier, not as an ordered date.

## Current status

The project is being rebuilt progressively around leakage-safe grouped
validation. The active foundation contains only repository-relative training
data access and validated binary-target construction. Grouped fold construction
and its audit are the next authorized component.

The former expanding-window methodology is preserved read-only under
`archive/v1/` at source commit
`e64aa6064e723bad935651bd0c664ea9c044ffc0`. Its notebooks, research modules,
training scripts, historical tests and submission log are not part of the
active V2 package.

## Application status

`app/`, `frontend/` and their tests remain active. They continue to serve the
locally generated `gradient_boosting_ret20_v1` artifact. This is an explicit
temporary compatibility state: the application has not yet been promoted to a
V2 model, and no V2 performance claim is made by the interface.

The local application still requires the ignored V1 files:

```text
models/gradient_boosting_ret20_v1.joblib
models/gradient_boosting_ret20_v1.metadata.json
```

See `docs/run_api_and_streamlit.md` for the FastAPI and Streamlit commands.

## Installation

`pyproject.toml` is the single source of dependency truth. Dependencies are
separated into:

- core: data handling and grouped validation;
- app: FastAPI, Streamlit and V1 artifact serving;
- research: notebook execution and plotting;
- dev: tests.

For the complete local development environment:

```powershell
python -m pip install -e '.[app,research,dev]'
```

`requirements.txt` is retained only as a generated compatibility entry point
and must not be edited manually.

## Data

Authorized challenge CSV files remain local and ignored under `data/raw/`.
The active package exposes loaders only for `X_train.csv` and `y_train.csv`.
Validation design and model selection must not access challenge test data.

## Repository layout

```text
archive/v1/             Read-only V1 research and modeling snapshot
app/                    Active FastAPI service, still serving the V1 model
frontend/               Active Streamlit client
research/references/    Read-only official benchmark reference
src/qrt_forecasting/    Installable active V2 package
tests/                  Active V2, application and frontend tests
```

## Guardrails

- `TS` is a group identifier only.
- All rows sharing a `TS` must stay together.
- V2 components are added only when an authorized phase needs them.
- The lockbox, model training, optimization and submissions are outside the
  current package-foundation step.
- Deep learning is outside the authorized scope.
