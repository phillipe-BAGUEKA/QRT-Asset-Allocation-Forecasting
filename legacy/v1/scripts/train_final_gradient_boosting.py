"""Train and serialize the frozen Gradient Boosting final model v1."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


MODEL_NAME = "gradient_boosting_ret20_v1"
TARGET_COLUMN = "class"
PREDICTION_THRESHOLD = 0.50
RANDOM_STATE = 42
SANITY_SAMPLE_SIZE = 1_024

FEATURE_COLUMNS = [f"RET_{index}" for index in range(1, 21)]

FINAL_MODEL_PARAMS = {
    "learning_rate": 0.05,
    "n_estimators": 50,
    "max_depth": 2,
    "min_samples_leaf": 20,
    "subsample": 0.7,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
}

VALIDATION_REFERENCE = {
    "protocol": "expanding_window_4_folds",
    "features": "RET_1_to_RET_20",
    "mlflow_parent_run_id": "b8976e8e2f6549eca140bb6f7cfd9538",
    "mean_valid_roc_auc": 0.5293628479202339,
    "fold_valid_roc_auc": [
        0.5198285092218297,
        0.5498522018953431,
        0.5235440312071667,
        0.5242266493565962,
    ],
    "std_valid_roc_auc": 0.011947347031439614,
    "worst_valid_roc_auc": 0.5198285092218297,
    "mean_train_roc_auc": 0.5341083034999283,
    "mean_roc_auc_gap": 0.004745455579694413,
    "mean_valid_log_loss": 0.6918455856929127,
}


def _run_git_command(repo_root: Path, *arguments: str) -> str:
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

    return completed_process.stdout.strip()


def _read_clean_git_metadata(repo_root: Path) -> tuple[str, str]:
    """Return commit and branch after requiring a clean worktree."""
    git_commit = _run_git_command(repo_root, "rev-parse", "HEAD")
    git_branch = _run_git_command(repo_root, "branch", "--show-current")
    worktree_status = _run_git_command(
        repo_root,
        "status",
        "--porcelain",
    )

    if not git_commit:
        raise ValueError("The current Git commit could not be determined.")

    if not git_branch:
        raise ValueError("The current Git branch could not be determined.")

    if worktree_status:
        raise ValueError(
            "The Git worktree must be clean before final-model training. "
            f"Detected changes:\n{worktree_status}"
        )

    return git_commit, git_branch


def _load_training_data(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load only QRT training features and targets."""
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

        return load_X_train(), load_y_train()
    finally:
        os.chdir(original_working_directory)


def _validate_raw_training_data(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> None:
    """Validate raw identifiers, dates, target, and required returns."""
    missing_X_columns = [
        column
        for column in ["ROW_ID", "TS", *FEATURE_COLUMNS]
        if column not in X_train.columns
    ]

    if missing_X_columns:
        raise ValueError(
            f"X_train is missing required columns: {missing_X_columns}."
        )

    missing_y_columns = [
        column
        for column in ("ROW_ID", "target")
        if column not in y_train.columns
    ]

    if missing_y_columns:
        raise ValueError(
            f"y_train is missing required columns: {missing_y_columns}."
        )

    target = y_train["target"]

    if target.isna().any():
        raise ValueError("y_train target contains missing values.")

    if not pd.api.types.is_numeric_dtype(target.dtype):
        raise ValueError("y_train target must have a numeric dtype.")

    if pd.api.types.is_bool_dtype(target.dtype):
        raise ValueError("y_train target must not have a boolean dtype.")

    if not np.isfinite(target.to_numpy()).all():
        raise ValueError("y_train target must contain only finite values.")

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


def _validate_joined_training_data(
    df_train: pd.DataFrame,
    expected_row_count: int,
) -> None:
    """Validate row preservation and the exact binary target domain."""
    if len(df_train) != expected_row_count:
        raise ValueError(
            "Joining X_train with y_train changed the number of rows: "
            f"expected {expected_row_count}, found {len(df_train)}."
        )

    if TARGET_COLUMN not in df_train.columns:
        raise ValueError(
            f"The joined training data has no '{TARGET_COLUMN}' column."
        )

    if df_train[TARGET_COLUMN].isna().any():
        raise ValueError(f"The '{TARGET_COLUMN}' column contains NaN.")

    observed_classes = set(df_train[TARGET_COLUMN].unique().tolist())

    if observed_classes != {0, 1}:
        raise ValueError(
            f"The '{TARGET_COLUMN}' classes must be exactly {{0, 1}}. "
            f"Observed values: {sorted(observed_classes)}."
        )


def _prepare_training_matrices(
    df_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select only the frozen ordered features and binary target."""
    expected_features = [f"RET_{index}" for index in range(1, 21)]

    if FEATURE_COLUMNS != expected_features:
        raise ValueError(
            "FEATURE_COLUMNS must be exactly RET_1 through RET_20 in "
            "ascending order."
        )

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in df_train.columns
    ]

    if missing_features:
        raise ValueError(
            f"The joined data is missing features: {missing_features}."
        )

    X = df_train[FEATURE_COLUMNS]
    y = df_train[TARGET_COLUMN]

    if X.columns.tolist() != expected_features:
        raise ValueError("The model feature order is not RET_1 to RET_20.")

    return X, y


def _build_final_model() -> Any:
    """Build the single frozen final-model pipeline."""
    from src.boosting_models import build_gradboosting_pipeline

    return build_gradboosting_pipeline(**FINAL_MODEL_PARAMS)


def _fit_model(model: Any, X: pd.DataFrame, y: pd.Series) -> float:
    """Fit exactly once and return the measured wall-clock duration."""
    fit_start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - fit_start


def _predict_and_validate(
    model: Any,
    sample: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated positive probabilities and thresholded classes."""
    probability_matrix = np.asarray(model.predict_proba(sample))

    if probability_matrix.ndim != 2 or probability_matrix.shape[1] < 2:
        raise ValueError("predict_proba must return at least two columns.")

    if probability_matrix.shape[0] != len(sample):
        raise ValueError(
            "predict_proba returned an incorrect number of rows."
        )

    probabilities = probability_matrix[:, 1]

    if len(probabilities) != len(sample):
        raise ValueError("The probability vector has an invalid length.")

    if not np.isfinite(probabilities).all():
        raise ValueError("The predicted probabilities must be finite.")

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("The predicted probabilities must lie in [0, 1].")

    predictions = (
        probabilities >= PREDICTION_THRESHOLD
    ).astype(int)

    if not set(np.unique(predictions).tolist()).issubset({0, 1}):
        raise ValueError("The thresholded predictions must be binary.")

    return probabilities, predictions


def _create_sanity_sample(X: pd.DataFrame) -> pd.DataFrame:
    """Select the first deterministic lightweight validation sample."""
    if X.empty:
        raise ValueError("Training features cannot be empty.")

    return X.iloc[: min(SANITY_SAMPLE_SIZE, len(X))]


def _atomic_dump_model(model: Any, model_path: Path) -> int:
    """Serialize a compressed model and atomically publish the artifact."""
    model_path = model_path.resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{model_path.name}.",
        suffix=".tmp",
        dir=model_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)

    try:
        joblib.dump(model, temporary_path, compress=3)

        if temporary_path.stat().st_size <= 0:
            raise ValueError("The temporary model artifact is empty.")

        os.replace(temporary_path, model_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise ValueError("The final model artifact is missing or empty.")

    return model_path.stat().st_size


def _verify_model_round_trip(
    model_path: Path,
    sample: pd.DataFrame,
    probabilities_before: np.ndarray,
    predictions_before: np.ndarray,
) -> Any:
    """Reload the artifact and verify strict prediction equivalence."""
    reloaded_model = joblib.load(model_path)

    if not hasattr(reloaded_model, "predict"):
        raise ValueError("The reloaded model does not provide predict.")

    if not hasattr(reloaded_model, "predict_proba"):
        raise ValueError(
            "The reloaded model does not provide predict_proba."
        )

    probabilities_after, predictions_after = _predict_and_validate(
        reloaded_model,
        sample,
    )
    np.testing.assert_allclose(
        probabilities_before,
        probabilities_after,
        rtol=1e-12,
        atol=1e-15,
    )

    if not np.array_equal(predictions_before, predictions_after):
        raise ValueError(
            "Binary predictions changed after the joblib round-trip."
        )

    return reloaded_model


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _build_metadata(
    *,
    git_commit: str,
    git_branch: str,
    training_row_count: int,
    training_date_count: int,
    training_start_date: str,
    training_end_date: str,
    training_positive_rate: float,
    fit_time_seconds: float,
    artifact_filename: str,
    artifact_size_bytes: int,
    artifact_sha256: str,
) -> dict[str, Any]:
    """Build JSON-safe provenance metadata for the final artifact."""
    metadata = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "model_family": "GradientBoostingClassifier",
        "pipeline": ["SimpleImputer", "GradientBoostingClassifier"],
        "selection_status": "stable_reference_not_exhaustively_tuned",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_count": len(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "model_parameters": dict(FINAL_MODEL_PARAMS),
        "training_row_count": int(training_row_count),
        "training_date_count": int(training_date_count),
        "training_start_date": str(training_start_date),
        "training_end_date": str(training_end_date),
        "training_positive_rate": float(training_positive_rate),
        "fit_time_seconds": float(fit_time_seconds),
        "artifact_filename": artifact_filename,
        "artifact_size_bytes": int(artifact_size_bytes),
        "artifact_sha256": artifact_sha256,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
        "validation_reference": dict(VALIDATION_REFERENCE),
    }
    json.dumps(metadata, allow_nan=False)
    return metadata


def _atomic_write_json(metadata: dict[str, Any], metadata_path: Path) -> None:
    """Atomically write UTF-8, strictly JSON-compliant metadata."""
    metadata_path = metadata_path.resolve()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
        dir=metadata_path.parent,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_handle:
            json.dump(
                metadata,
                file_handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            file_handle.write("\n")
            file_handle.flush()
            os.fsync(file_handle.fileno())

        os.replace(temporary_path, metadata_path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _print_training_summary(
    *,
    model_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    probabilities: np.ndarray,
) -> None:
    """Print the final training and serialization summary."""
    print("\nFinal model training completed successfully.")
    print(f"model_name: {metadata['model_name']}")
    print(f"model_path: {model_path}")
    print(f"metadata_path: {metadata_path}")
    print(f"training_row_count: {metadata['training_row_count']}")
    print(f"training_date_count: {metadata['training_date_count']}")
    print(
        "training_period: "
        f"{metadata['training_start_date']} -> "
        f"{metadata['training_end_date']}"
    )
    print(f"feature_columns: {metadata['feature_columns']}")
    print(
        "training_positive_rate: "
        f"{metadata['training_positive_rate']}"
    )
    print(f"fit_time_seconds: {metadata['fit_time_seconds']}")
    print(f"artifact_size_bytes: {metadata['artifact_size_bytes']}")
    print(f"artifact_sha256: {metadata['artifact_sha256']}")
    print(f"sanity_probability_mean: {float(np.mean(probabilities))}")
    print(f"sanity_probability_std: {float(np.std(probabilities))}")
    print("joblib_round_trip: verified")
    print(f"git_commit: {metadata['git_commit']}")
    print(f"git_branch: {metadata['git_branch']}")


def main() -> None:
    """Train, validate, serialize, and document the frozen final model."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_text = str(repo_root)

    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)

    git_commit, git_branch = _read_clean_git_metadata(repo_root)
    X_train, y_train = _load_training_data(repo_root)
    _validate_raw_training_data(X_train, y_train)

    expected_row_count = len(X_train)

    from src.target import create_class_column

    df_train, enriched_y_train = create_class_column(X_train, y_train)
    _validate_joined_training_data(df_train, expected_row_count)

    del X_train
    del y_train
    del enriched_y_train
    gc.collect()

    X, y = _prepare_training_matrices(df_train)
    training_row_count = len(df_train)
    training_date_count = int(df_train["TS"].nunique())
    training_start_date = str(df_train["TS"].min())
    training_end_date = str(df_train["TS"].max())
    training_positive_rate = float(y.mean())

    del df_train
    gc.collect()

    model = _build_final_model()
    fit_time_seconds = _fit_model(model, X, y)
    sample = _create_sanity_sample(X)
    probabilities_before, predictions_before = _predict_and_validate(
        model,
        sample,
    )

    model_path = (
        repo_root / "models" / f"{MODEL_NAME}.joblib"
    ).resolve()
    metadata_path = (
        repo_root / "models" / f"{MODEL_NAME}.metadata.json"
    ).resolve()
    artifact_size_bytes = _atomic_dump_model(model, model_path)
    _verify_model_round_trip(
        model_path,
        sample,
        probabilities_before,
        predictions_before,
    )
    artifact_sha256 = _sha256_file(model_path)
    metadata = _build_metadata(
        git_commit=git_commit,
        git_branch=git_branch,
        training_row_count=training_row_count,
        training_date_count=training_date_count,
        training_start_date=training_start_date,
        training_end_date=training_end_date,
        training_positive_rate=training_positive_rate,
        fit_time_seconds=fit_time_seconds,
        artifact_filename=model_path.name,
        artifact_size_bytes=artifact_size_bytes,
        artifact_sha256=artifact_sha256,
    )
    _atomic_write_json(metadata, metadata_path)
    _print_training_summary(
        model_path=model_path,
        metadata_path=metadata_path,
        metadata=metadata,
        probabilities=probabilities_before,
    )


if __name__ == "__main__":
    main()
