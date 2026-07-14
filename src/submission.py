"""
Generate an official submission for the QRT Asset Allocation challenge.

This module loads the raw datasets, creates the binary target, applies
the selected feature-engineering transformations, trains the final model
on the complete training dataset and exports validated test predictions.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from src.advanced_boosting import build_xgboost_classifier
from src.data_loading import load_X_test, load_X_train, load_y_train
from src.features import (
    create_momentum_features,
    create_momentum_volatility_ratio_features,
    create_volatility_features,
    create_volume_features,
)
from src.target import create_class_column


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIRECTORY = PROJECT_ROOT / "data" / "submissions"

SUBMISSION_FILENAME = "submission_v004.csv"
TARGET_COLUMN = "class"

EXCLUDED_FEATURE_COLUMNS = {
    "ROW_ID",
    "TS",
    "ALLOCATION",
    "target",
    "class",
}


def create_all_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all retained numerical feature-engineering transformations.

    The same function must be applied independently to the training and
    official test datasets to guarantee identical feature definitions.

    Args:
        df:
            Raw feature DataFrame containing historical return and signed
            volume columns.

    Returns:
        A copy of the input DataFrame enriched with momentum, volatility,
        risk-adjusted momentum and volume features.
    """
    df = df.copy()

    df = create_momentum_features(df)
    df = create_volatility_features(df)
    df = create_momentum_volatility_ratio_features(df)
    df = create_volume_features(df)

    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numerical model features while excluding identifiers and targets.

    Args:
        df:
            Engineered training DataFrame.

    Returns:
        Ordered list of numerical columns retained for model training.

    Raises:
        ValueError:
            If no usable numerical feature is found.
    """
    numerical_columns = df.select_dtypes(include="number").columns

    feature_columns = [
        column
        for column in numerical_columns
        if column not in EXCLUDED_FEATURE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No numerical feature was selected.")

    return feature_columns


def validate_train_test_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """
    Verify that every selected training feature is available in the test set.

    Args:
        df_train:
            Engineered training DataFrame.

        df_test:
            Engineered official test DataFrame.

        feature_columns:
            Ordered list of variables selected from the training dataset.

    Raises:
        ValueError:
            If features are missing, duplicated or contain infinite values.
    """
    missing_train_columns = [
        column for column in feature_columns
        if column not in df_train.columns
    ]

    missing_test_columns = [
        column for column in feature_columns
        if column not in df_test.columns
    ]

    duplicated_features = pd.Index(feature_columns)[
        pd.Index(feature_columns).duplicated()
    ].tolist()

    if missing_train_columns:
        raise ValueError(
            f"Missing training features: {missing_train_columns}"
        )

    if missing_test_columns:
        raise ValueError(
            f"Missing test features: {missing_test_columns}"
        )

    if duplicated_features:
        raise ValueError(
            f"Duplicated feature names: {duplicated_features}"
        )


def build_final_model() -> Any:
    """
    Build the final model retained for the current submission experiment.

    Returns:
        Untrained Scikit-Learn compatible prediction pipeline.
    """
    return build_xgboost_classifier()


def build_submission(
    row_ids: pd.Series,
    predictions: pd.Series | list | Any,
) -> pd.DataFrame:
    """
    Build the official submission DataFrame.

    Args:
        row_ids:
            Test observation identifiers.

        predictions:
            Binary model predictions in the same order as ``row_ids``.

    Returns:
        Submission DataFrame containing ``ROW_ID`` and ``prediction``.

    Raises:
        ValueError:
            If the number of predictions differs from the number of IDs.
    """
    if len(row_ids) != len(predictions):
        raise ValueError(
            "The number of predictions differs from the number of ROW_IDs."
        )

    submission = pd.DataFrame(
        {
            "ROW_ID": row_ids.to_numpy(),
            "prediction": predictions,
        }
    )

    return submission


def validate_submission_format(
    submission: pd.DataFrame,
    expected_row_ids: pd.Series,
) -> None:
    """
    Validate the final submission structure and prediction values.

    Args:
        submission:
            DataFrame expected to contain ``ROW_ID`` and ``prediction``.

        expected_row_ids:
            Original test identifiers used to verify row preservation.

    Raises:
        ValueError:
            If columns, identifiers, row order or predictions are invalid.
    """
    expected_columns = ["ROW_ID", "prediction"]

    if submission.columns.tolist() != expected_columns:
        raise ValueError(
            f"Submission columns must be exactly {expected_columns}."
        )

    if len(submission) != len(expected_row_ids):
        raise ValueError(
            "Submission row count differs from the test row count."
        )

    if submission["ROW_ID"].isna().any():
        raise ValueError("ROW_ID contains missing values.")

    if submission["ROW_ID"].duplicated().any():
        raise ValueError("ROW_ID contains duplicated values.")

    if not submission["ROW_ID"].reset_index(drop=True).equals(
        expected_row_ids.reset_index(drop=True)
    ):
        raise ValueError(
            "The submission changed the order of the test observations."
        )

    if submission["prediction"].isna().any():
        raise ValueError("Predictions contain missing values.")

    prediction_values = set(submission["prediction"].unique())

    if not prediction_values.issubset({0, 1}):
        raise ValueError(
            f"Predictions must be binary. Found: {prediction_values}"
        )


def export_submission_csv(
    submission: pd.DataFrame,
    filename: str = SUBMISSION_FILENAME,
) -> Path:
    """
    Export a validated submission to the submissions directory.

    Args:
        submission:
            Validated submission DataFrame.

        filename:
            Output CSV filename.

    Returns:
        Path of the generated CSV file.
    """
    if not filename.lower().endswith(".csv"):
        raise ValueError("The submission filename must end with '.csv'.")

    SUBMISSION_DIRECTORY.mkdir(parents=True, exist_ok=True)

    output_path = SUBMISSION_DIRECTORY / filename
    submission.to_csv(output_path, index=False)

    return output_path


def main() -> None:
    """
    Execute the complete final-training and submission workflow.
    """
    print("Loading datasets...")

    X_train = load_X_train()
    X_test = load_X_test()
    y_train = load_y_train()

    original_test_row_ids = X_test["ROW_ID"].copy()

    print("Creating binary target...")

    df_train, _ = create_class_column(X_train, y_train)

    print("Creating engineered features...")

    df_train = create_all_engineered_features(df_train)
    df_test = create_all_engineered_features(X_test)

    feature_columns = select_feature_columns(df_train)

    validate_train_test_features(
        df_train=df_train,
        df_test=df_test,
        feature_columns=feature_columns,
    )

    print(f"Number of selected features: {len(feature_columns)}")

    model = build_final_model()

    print("Training final model...")

    model.fit(
        df_train[feature_columns],
        df_train[TARGET_COLUMN],
    )

    print("Generating predictions...")

    predictions = model.predict(df_test[feature_columns])

    submission = build_submission(
        row_ids=original_test_row_ids,
        predictions=predictions,
    )

    validate_submission_format(
        submission=submission,
        expected_row_ids=original_test_row_ids,
    )

    output_path = export_submission_csv(submission)

    positive_rate = submission["prediction"].mean()

    print("Submission successfully generated.")
    print(f"Output path: {output_path}")
    print(f"Rows: {len(submission):,}")
    print(f"Positive predictions: {submission['prediction'].sum():,}")
    print(f"Predicted positive rate: {positive_rate:.2%}")


if __name__ == "__main__":
    main()