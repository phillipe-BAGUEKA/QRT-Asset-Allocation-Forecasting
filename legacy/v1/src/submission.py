"""
Generate an official CNN 1D submission for the QRT Asset Allocation challenge.

The selected model is a one-dimensional convolutional neural network trained
on the ordered return history RET_1...RET_20. The script preserves the
official test row order, performs strict input checks, exports binary
predictions and saves a JSON file describing the experiment.

The CNN is the preferred neural-network candidate because the return columns
form a short ordered sequence. It can learn local temporal motifs such as
momentum, reversals and alternating signs while remaining simpler and easier
to regularise than an LSTM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loading import load_X_test, load_X_train, load_y_train
from neural_networks import build_cnn1d_classifier
from target import create_class_column


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIRECTORY = PROJECT_ROOT / "data" / "submissions"

SUBMISSION_FILENAME = "submission_v007_cnn1d.csv"
TARGET_COLUMN = "class"
ROW_ID_COLUMN = "ROW_ID"
TIMESTAMP_COLUMN = "TS"
PREDICTION_COLUMN = "prediction"
PROBABILITY_COLUMN = "probability"

RETURN_FEATURE_COLUMNS = [
    f"RET_{lag}"
    for lag in range(1, 21)
]

PREDICTION_THRESHOLD = 0.50

MODEL_PARAMETERS = {
    "conv_channels": (32, 64),
    "kernel_size": 3,
    "dense_dim": 64,
    "dropout": 0.25,
    "learning_rate": 5e-4,
    "weight_decay": 1e-4,
    "batch_size": 1024,
    "max_epochs": 75,
    "patience": 8,
    "validation_fraction": 0.15,
    "reverse_sequence": True,
    "random_state": 42,
    "device": "auto",
    "verbose": False,
}


def validate_raw_datasets(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
) -> None:
    """
    Validate the raw train, test and target datasets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Raw training features.

    X_test : pd.DataFrame
        Raw official test features.

    y_train : pd.DataFrame or pd.Series
        Raw training target data.

    Raises
    ------
    TypeError
        If feature inputs are not pandas DataFrames.

    ValueError
        If required identifiers, timestamps or return features are missing,
        duplicated or inconsistent.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    required_columns = {
        ROW_ID_COLUMN,
        TIMESTAMP_COLUMN,
        *RETURN_FEATURE_COLUMNS,
    }

    missing_train_columns = sorted(
        required_columns.difference(X_train.columns)
    )
    missing_test_columns = sorted(
        required_columns.difference(X_test.columns)
    )

    if missing_train_columns:
        raise ValueError(
            f"Missing training columns: {missing_train_columns}"
        )

    if missing_test_columns:
        raise ValueError(
            f"Missing test columns: {missing_test_columns}"
        )

    if X_train.columns.duplicated().any():
        duplicated = X_train.columns[
            X_train.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Duplicated training columns: {duplicated}"
        )

    if X_test.columns.duplicated().any():
        duplicated = X_test.columns[
            X_test.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Duplicated test columns: {duplicated}"
        )

    if X_train[ROW_ID_COLUMN].isna().any():
        raise ValueError("Training ROW_ID contains missing values.")

    if X_test[ROW_ID_COLUMN].isna().any():
        raise ValueError("Test ROW_ID contains missing values.")

    if X_train[ROW_ID_COLUMN].duplicated().any():
        raise ValueError("Training ROW_ID contains duplicates.")

    if X_test[ROW_ID_COLUMN].duplicated().any():
        raise ValueError("Test ROW_ID contains duplicates.")

    if X_train[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Training TS contains missing values.")

    if X_test[TIMESTAMP_COLUMN].isna().any():
        raise ValueError("Test TS contains missing values.")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train do not contain the same number of rows."
        )


def validate_model_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Validate the exact numerical features supplied to the CNN.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame containing the binary target.

    df_test : pd.DataFrame
        Official test DataFrame.

    feature_columns : list[str]
        Ordered return columns used by the model.

    Raises
    ------
    ValueError
        If columns are missing, duplicated, non-numeric or entirely missing.
    """
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")

    duplicated_features = pd.Index(feature_columns)[
        pd.Index(feature_columns).duplicated()
    ].tolist()

    if duplicated_features:
        raise ValueError(
            f"Duplicated feature names: {duplicated_features}"
        )

    missing_train_features = [
        column
        for column in feature_columns
        if column not in df_train.columns
    ]
    missing_test_features = [
        column
        for column in feature_columns
        if column not in df_test.columns
    ]

    if missing_train_features:
        raise ValueError(
            f"Missing training features: {missing_train_features}"
        )

    if missing_test_features:
        raise ValueError(
            f"Missing test features: {missing_test_features}"
        )

    non_numeric_train = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df_train[column])
    ]
    non_numeric_test = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df_test[column])
    ]

    if non_numeric_train:
        raise ValueError(
            f"Non-numeric training features: {non_numeric_train}"
        )

    if non_numeric_test:
        raise ValueError(
            f"Non-numeric test features: {non_numeric_test}"
        )

    entirely_missing_train = [
        column
        for column in feature_columns
        if df_train[column].isna().all()
    ]
    entirely_missing_test = [
        column
        for column in feature_columns
        if df_test[column].isna().all()
    ]

    if entirely_missing_train:
        raise ValueError(
            "Training features containing only missing values: "
            f"{entirely_missing_train}"
        )

    if entirely_missing_test:
        raise ValueError(
            "Test features containing only missing values: "
            f"{entirely_missing_test}"
        )


def prepare_training_dataframe(
    df_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Sort training observations chronologically for internal early stopping.

    A stable sort keeps the original order within each date. This matters
    because the neural estimator uses the final portion of the supplied
    training data as its internal validation period.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame containing ``TS`` and the binary target.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted training DataFrame.
    """
    return (
        df_train
        .sort_values(
            by=[TIMESTAMP_COLUMN, ROW_ID_COLUMN],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def build_final_model() -> Any:
    """
    Build the CNN 1D retained for the official neural-network submission.

    Returns
    -------
    Any
        Untrained Scikit-Learn compatible PyTorch classifier.
    """
    return build_cnn1d_classifier(**MODEL_PARAMETERS)


def build_submission(
    row_ids: pd.Series,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Build the official binary submission DataFrame.

    Parameters
    ----------
    row_ids : pd.Series
        Test observation identifiers in official order.

    predictions : np.ndarray
        Binary predictions in the same order as ``row_ids``.

    Returns
    -------
    pd.DataFrame
        Submission containing ``ROW_ID`` and ``prediction``.

    Raises
    ------
    ValueError
        If identifiers and predictions have different lengths.
    """
    predictions = np.asarray(predictions).reshape(-1)

    if len(row_ids) != len(predictions):
        raise ValueError(
            "The number of predictions differs from the number of ROW_IDs."
        )

    return pd.DataFrame(
        {
            ROW_ID_COLUMN: row_ids.to_numpy(),
            PREDICTION_COLUMN: predictions.astype(int),
        }
    )


def validate_probabilities(
    probabilities: np.ndarray,
    expected_length: int,
) -> None:
    """
    Validate predicted probabilities before thresholding.

    Parameters
    ----------
    probabilities : np.ndarray
        Positive-class probabilities.

    expected_length : int
        Expected number of test observations.

    Raises
    ------
    ValueError
        If probabilities have an invalid length or contain invalid values.
    """
    probabilities = np.asarray(probabilities).reshape(-1)

    if len(probabilities) != expected_length:
        raise ValueError(
            "The number of probabilities differs from the test row count."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Predicted probabilities contain NaN or infinite values."
        )

    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(
            "Predicted probabilities must lie in [0, 1]."
        )


def validate_submission_format(
    submission: pd.DataFrame,
    expected_row_ids: pd.Series,
) -> None:
    """
    Validate the official submission structure and values.

    Parameters
    ----------
    submission : pd.DataFrame
        Submission expected to contain ``ROW_ID`` and ``prediction``.

    expected_row_ids : pd.Series
        Original test identifiers used to verify row preservation.

    Raises
    ------
    ValueError
        If columns, identifiers, ordering or predictions are invalid.
    """
    expected_columns = [
        ROW_ID_COLUMN,
        PREDICTION_COLUMN,
    ]

    if submission.columns.tolist() != expected_columns:
        raise ValueError(
            f"Submission columns must be exactly {expected_columns}."
        )

    if len(submission) != len(expected_row_ids):
        raise ValueError(
            "Submission row count differs from the test row count."
        )

    if submission[ROW_ID_COLUMN].isna().any():
        raise ValueError("ROW_ID contains missing values.")

    if submission[ROW_ID_COLUMN].duplicated().any():
        raise ValueError("ROW_ID contains duplicated values.")

    if not submission[ROW_ID_COLUMN].reset_index(drop=True).equals(
        expected_row_ids.reset_index(drop=True)
    ):
        raise ValueError(
            "The submission changed the official test row order."
        )

    if submission[PREDICTION_COLUMN].isna().any():
        raise ValueError("Predictions contain missing values.")

    prediction_values = set(
        submission[PREDICTION_COLUMN].unique()
    )

    if not prediction_values.issubset({0, 1}):
        raise ValueError(
            f"Predictions must be binary. Found: {prediction_values}"
        )


def export_submission_csv(
    submission: pd.DataFrame,
    filename: str = SUBMISSION_FILENAME,
) -> Path:
    """
    Export the validated submission to the submissions directory.

    Parameters
    ----------
    submission : pd.DataFrame
        Validated submission DataFrame.

    filename : str
        Output CSV filename.

    Returns
    -------
    Path
        Path of the generated submission file.
    """
    if not filename.lower().endswith(".csv"):
        raise ValueError(
            "The submission filename must end with '.csv'."
        )

    SUBMISSION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = SUBMISSION_DIRECTORY / filename
    submission.to_csv(output_path, index=False)

    return output_path


def export_probability_diagnostics(
    row_ids: pd.Series,
    probabilities: np.ndarray,
    submission_path: Path,
) -> Path:
    """
    Export test probabilities for local diagnostics and future blending.

    This file is not the official challenge submission.

    Parameters
    ----------
    row_ids : pd.Series
        Official test identifiers.

    probabilities : np.ndarray
        Positive-class probabilities.

    submission_path : Path
        Path of the official binary submission.

    Returns
    -------
    Path
        Path of the probability CSV file.
    """
    probability_path = submission_path.with_name(
        f"{submission_path.stem}_probabilities.csv"
    )

    probability_df = pd.DataFrame(
        {
            ROW_ID_COLUMN: row_ids.to_numpy(),
            PROBABILITY_COLUMN: np.asarray(
                probabilities
            ).reshape(-1),
        }
    )

    probability_df.to_csv(
        probability_path,
        index=False,
    )

    return probability_path


def export_experiment_metadata(
    submission_path: Path,
    model: Any,
    train_positive_rate: float,
    predicted_positive_rate: float,
    probability_mean: float,
    probability_std: float,
) -> Path:
    """
    Save the experiment configuration beside the submission.

    Parameters
    ----------
    submission_path : Path
        Generated official submission path.

    model : Any
        Trained neural estimator.

    train_positive_rate : float
        Positive-class rate in the training target.

    predicted_positive_rate : float
        Positive-class rate in binary test predictions.

    probability_mean : float
        Mean positive-class probability on the test set.

    probability_std : float
        Standard deviation of positive-class probabilities.

    Returns
    -------
    Path
        Path of the generated JSON metadata file.
    """
    metadata_path = submission_path.with_suffix(".json")

    metadata = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "submission_file": submission_path.name,
        "model": "cnn1d",
        "features": RETURN_FEATURE_COLUMNS,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "model_parameters": {
            key: (
                list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in MODEL_PARAMETERS.items()
        },
        "selected_epoch_count": int(
            getattr(model, "best_epoch_", -1)
        ),
        "train_positive_rate": train_positive_rate,
        "predicted_positive_rate": predicted_positive_rate,
        "test_probability_mean": probability_mean,
        "test_probability_std": probability_std,
    }

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return metadata_path


def main() -> None:
    """
    Execute the complete CNN training and submission workflow.
    """
    print("Loading datasets...")

    X_train = load_X_train()
    X_test = load_X_test()
    y_train = load_y_train()

    validate_raw_datasets(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
    )

    original_test_row_ids = X_test[
        ROW_ID_COLUMN
    ].copy()

    print("Creating binary target...")

    df_train, _ = create_class_column(
        X_train,
        y_train,
    )

    observed_classes = set(
        df_train[TARGET_COLUMN].dropna().unique()
    )

    if observed_classes != {0, 1}:
        raise ValueError(
            "The generated target must contain exactly classes 0 and 1. "
            f"Found: {observed_classes}"
        )

    df_train = prepare_training_dataframe(
        df_train
    )

    feature_columns = RETURN_FEATURE_COLUMNS.copy()

    validate_model_features(
        df_train=df_train,
        df_test=X_test,
        feature_columns=feature_columns,
    )

    model = build_final_model()

    print("Selected model: CNN 1D")
    print(
        "Feature order supplied to the model: "
        "RET_1 ... RET_20"
    )
    print(
        "Chronological order used internally: "
        "RET_20 ... RET_1"
    )
    print(
        f"Number of selected features: "
        f"{len(feature_columns)}"
    )

    print("Training final CNN 1D model...")

    model.fit(
        df_train[feature_columns],
        df_train[TARGET_COLUMN],
    )

    selected_epochs = getattr(
        model,
        "best_epoch_",
        None,
    )

    if selected_epochs is not None:
        print(
            f"Selected epoch count: "
            f"{selected_epochs}"
        )

    print("Generating test probabilities...")

    probabilities = model.predict_proba(
        X_test[feature_columns]
    )[:, 1]

    validate_probabilities(
        probabilities=probabilities,
        expected_length=len(X_test),
    )

    predictions = (
        probabilities >= PREDICTION_THRESHOLD
    ).astype(int)

    submission = build_submission(
        row_ids=original_test_row_ids,
        predictions=predictions,
    )

    validate_submission_format(
        submission=submission,
        expected_row_ids=original_test_row_ids,
    )

    submission_path = export_submission_csv(
        submission
    )

    probability_path = export_probability_diagnostics(
        row_ids=original_test_row_ids,
        probabilities=probabilities,
        submission_path=submission_path,
    )

    train_positive_rate = float(
        df_train[TARGET_COLUMN].mean()
    )
    predicted_positive_rate = float(
        submission[PREDICTION_COLUMN].mean()
    )
    probability_mean = float(
        np.mean(probabilities)
    )
    probability_std = float(
        np.std(probabilities)
    )

    metadata_path = export_experiment_metadata(
        submission_path=submission_path,
        model=model,
        train_positive_rate=train_positive_rate,
        predicted_positive_rate=predicted_positive_rate,
        probability_mean=probability_mean,
        probability_std=probability_std,
    )

    print("Submission successfully generated.")
    print(f"Official submission: {submission_path}")
    print(f"Probability diagnostics: {probability_path}")
    print(f"Experiment metadata: {metadata_path}")
    print(f"Rows: {len(submission):,}")
    print(
        "Positive predictions: "
        f"{submission[PREDICTION_COLUMN].sum():,}"
    )
    print(
        "Training positive rate: "
        f"{train_positive_rate:.2%}"
    )
    print(
        "Predicted positive rate: "
        f"{predicted_positive_rate:.2%}"
    )
    print(
        "Mean test probability: "
        f"{probability_mean:.6f}"
    )
    print(
        "Test probability standard deviation: "
        f"{probability_std:.6f}"
    )


if __name__ == "__main__":
    main()