"""Run one controlled Gradient Boosting Optuna trial with MLflow tracking."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


RANDOM_STATE = 42
N_FOLDS = 4
VALIDATION_SIZE = 120
N_TRIALS = 1
TIMEOUT_SECONDS = 900.0

EXPERIMENT_NAME = "qrt_gradient_boosting_optuna"
PARENT_RUN_NAME = "smoke_test_1_trial"
FEATURE_SET_NAME = "RET_1_to_RET_20"
PROJECT_STAGE = "real_integration_smoke_test"

EXPECTED_FEATURE_COLUMNS = [f"RET_{index}" for index in range(1, 21)]

REFERENCE_TRIAL_PARAMS = {
    "learning_rate": 0.05,
    "n_estimators": 50,
    "max_depth": 2,
    "min_samples_leaf": 20,
    "subsample": 0.7,
    "max_features": "sqrt",
}


def _run_git_command(
    repo_root: Path,
    *arguments: str,
    require_output: bool = True,
) -> str:
    """Run one read-only Git command and return its stripped output."""
    try:
        completed_process = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        command = "git " + " ".join(arguments)
        raise ValueError(
            f"Unable to determine Git metadata with '{command}'."
        ) from error

    output = completed_process.stdout.strip()

    if require_output and not output:
        command = "git " + " ".join(arguments)
        raise ValueError(
            f"Git command '{command}' returned no usable value."
        )

    return output


def _read_clean_git_metadata(repo_root: Path) -> tuple[str, str]:
    """Return the current commit and branch after requiring a clean tree."""
    git_commit = _run_git_command(repo_root, "rev-parse", "HEAD")
    git_branch = _run_git_command(repo_root, "branch", "--show-current")
    worktree_status = _run_git_command(
        repo_root,
        "status",
        "--porcelain",
        require_output=False,
    )

    if worktree_status:
        raise ValueError(
            "The Git worktree must be clean before running the smoke test. "
            f"Detected changes:\n{worktree_status}"
        )

    return git_commit, git_branch


def _validate_raw_training_features(
    X_train: Any,
    y_train: Any,
) -> None:
    """Validate identifiers and return columns before target creation."""
    missing_identifier_columns = [
        column
        for column in ("ROW_ID", "TS")
        if column not in X_train.columns
    ]

    if missing_identifier_columns:
        raise ValueError(
            "X_train is missing required identifier columns: "
            f"{missing_identifier_columns}."
        )

    if "ROW_ID" not in y_train.columns:
        raise ValueError("y_train is missing required column 'ROW_ID'.")

    if X_train["ROW_ID"].isna().any():
        raise ValueError("X_train contains missing ROW_ID values.")

    if X_train["ROW_ID"].duplicated().any():
        raise ValueError("X_train contains duplicate ROW_ID values.")

    if y_train["ROW_ID"].isna().any():
        raise ValueError("y_train contains missing ROW_ID values.")

    if y_train["ROW_ID"].duplicated().any():
        raise ValueError("y_train contains duplicate ROW_ID values.")

    X_train_row_ids = set(X_train["ROW_ID"].tolist())
    y_train_row_ids = set(y_train["ROW_ID"].tolist())

    if X_train_row_ids != y_train_row_ids:
        raise ValueError(
            "X_train and y_train must contain exactly the same ROW_ID "
            "values."
        )

    missing_return_columns = [
        column
        for column in EXPECTED_FEATURE_COLUMNS
        if column not in X_train.columns
    ]

    if missing_return_columns:
        raise ValueError(
            "X_train is missing required return columns: "
            f"{missing_return_columns}."
        )


def _validate_joined_training_data(
    df_train: Any,
    expected_row_count: int,
) -> None:
    """Validate row preservation and the binary classification target."""
    if len(df_train) != expected_row_count:
        raise ValueError(
            "Joining X_train with y_train changed the number of rows: "
            f"expected {expected_row_count}, found {len(df_train)}."
        )

    if "class" not in df_train.columns:
        raise ValueError("The joined training data has no 'class' column.")

    if df_train["class"].isna().any():
        raise ValueError("The 'class' column contains missing values.")

    observed_classes = set(df_train["class"].unique().tolist())

    if not observed_classes.issubset({0, 1}):
        raise ValueError(
            "The 'class' column must contain only 0 and 1. "
            f"Observed values: {sorted(observed_classes)}."
        )


def _build_mlflow_server_command(
    repo_root: Path,
    tracking_db_path: Path,
) -> str:
    """Build a PowerShell command for the local MLflow server."""
    mlflow_executable = (
        repo_root / ".venv" / "Scripts" / "mlflow.exe"
    ).resolve()
    tracking_uri = f"sqlite:///{tracking_db_path.as_posix()}"

    return (
        f'& "{mlflow_executable}" server '
        f'--backend-store-uri "{tracking_uri}" '
        "--port 5000"
    )


def _print_run_summary(
    *,
    parent_run_id: str,
    trial: Any,
    tracking_db_path: Path,
    artifact_root: Path,
    mlflow_server_command: str,
) -> None:
    """Print the identifiers, results, diagnostics, and local paths."""
    print("\nSmoke test completed successfully.")
    print(f"parent_run_id: {parent_run_id}")
    print(f"trial_number: {trial.number}")
    print(f"trial_state: {trial.state.name}")
    print(f"objective_value: {trial.value}")
    print(
        "parameters: "
        + json.dumps(trial.params, indent=2, sort_keys=True)
    )
    print(
        "user_attrs: "
        + json.dumps(trial.user_attrs, indent=2, sort_keys=True)
    )
    print(
        "total_fit_time_seconds: "
        f"{trial.user_attrs.get('total_fit_time_seconds')}"
    )
    print(f"mlflow_database: {tracking_db_path}")
    print(f"mlflow_artifacts: {artifact_root}")
    print(f"mlflow_server_command: {mlflow_server_command}")
    print(
        "Note: timeout=900.0 is not a hard timeout for a trial that "
        "has already started."
    )


def main() -> None:
    """Run the single-trial real-data integration smoke test."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_text = str(repo_root)

    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)

    git_commit, git_branch = _read_clean_git_metadata(repo_root)

    original_working_directory = Path.cwd()
    notebooks_directory = repo_root / "notebooks"

    if not notebooks_directory.is_dir():
        raise ValueError(
            f"Notebook directory does not exist: {notebooks_directory}."
        )

    try:
        os.chdir(notebooks_directory)

        # Temporary workaround: data_loading.py currently derives paths from
        # Path.cwd().parent. It should later derive them from its own __file__.
        from src.data_loading import load_X_train, load_y_train

        X_train = load_X_train()
        y_train = load_y_train()
    finally:
        os.chdir(original_working_directory)

    from optuna.pruners import NopPruner

    from src.features import ret_features
    from src.mlflow_tracking import (
        configure_local_mlflow,
        run_optuna_study_with_mlflow,
    )
    from src.optimization import (
        create_gradient_boosting_objective,
        create_gradient_boosting_study,
    )
    from src.target import create_class_column
    from src.validation import (
        check_temporal_folds,
        create_expanding_window_folds,
    )

    _validate_raw_training_features(X_train, y_train)
    expected_row_count = len(X_train)
    df_train, enriched_y_train = create_class_column(X_train, y_train)
    _validate_joined_training_data(df_train, expected_row_count)

    del X_train
    del y_train
    del enriched_y_train
    gc.collect()

    dates = sorted(df_train["TS"].unique().tolist())
    folds = create_expanding_window_folds(
        dates,
        size=VALIDATION_SIZE,
        k=N_FOLDS,
    )

    if len(folds) != N_FOLDS:
        raise ValueError(
            f"Expected {N_FOLDS} temporal folds, found {len(folds)}."
        )

    folds_are_valid = check_temporal_folds(
        folds,
        dates,
        validation_size=VALIDATION_SIZE,
    )

    if folds_are_valid is not True:
        raise ValueError(
            "check_temporal_folds did not explicitly return True."
        )

    feature_cols = ret_features(df_train)

    if feature_cols != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(
            "The feature list must be exactly RET_1 through RET_20 in "
            f"ascending order. Found: {feature_cols}."
        )

    objective = create_gradient_boosting_objective(
        df=df_train,
        folds=folds,
        feature_cols=feature_cols,
        target_col="class",
        random_state=RANDOM_STATE,
    )
    study = create_gradient_boosting_study(seed=RANDOM_STATE)

    if not isinstance(study.pruner, NopPruner):
        raise ValueError(
            "The smoke-test Study must use Optuna's NopPruner."
        )

    study.enqueue_trial(REFERENCE_TRIAL_PARAMS)

    tracking_db_path = (
        repo_root / "mlflow_data" / "tracking" / "mlflow.db"
    ).resolve()
    artifact_root = (
        repo_root / "mlflow_data" / "artifacts"
    ).resolve()
    experiment_id = configure_local_mlflow(
        tracking_db_path=tracking_db_path,
        artifact_root=artifact_root,
        experiment_name=EXPERIMENT_NAME,
    )

    print(
        "Starting one real-data trial. timeout=900.0 does not interrupt "
        "a trial that has already started."
    )

    returned_study, parent_run_id = run_optuna_study_with_mlflow(
        study=study,
        objective=objective,
        experiment_id=experiment_id,
        parent_run_name=PARENT_RUN_NAME,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        parent_params={
            "n_folds": N_FOLDS,
            "sampler_seed": RANDOM_STATE,
            "random_state": RANDOM_STATE,
        },
        parent_tags={
            "feature_set": FEATURE_SET_NAME,
            "project_stage": PROJECT_STAGE,
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_worktree_status": "clean",
        },
        fixed_trial_params={"random_state": RANDOM_STATE},
        child_tags={
            "feature_set": FEATURE_SET_NAME,
            "project_stage": PROJECT_STAGE,
        },
    )

    if len(returned_study.trials) != N_TRIALS:
        raise ValueError(
            f"Expected exactly {N_TRIALS} trial, found "
            f"{len(returned_study.trials)}."
        )

    trial = returned_study.trials[0]
    mlflow_server_command = _build_mlflow_server_command(
        repo_root,
        tracking_db_path,
    )
    _print_run_summary(
        parent_run_id=parent_run_id,
        trial=trial,
        tracking_db_path=tracking_db_path,
        artifact_root=artifact_root,
        mlflow_server_command=mlflow_server_command,
    )


if __name__ == "__main__":
    main()
