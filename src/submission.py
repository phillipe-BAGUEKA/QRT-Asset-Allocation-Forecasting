from pathlib import Path

import pandas as pd

from data_loading import load_X_train, load_X_test, load_y_train
from features import ret_features
from modeling import build_logistic_regression_pipeline
from target import create_class_column


SUBMISSION_PATH = Path.cwd().parent / "data" / "submissions"


def _validate_submission_format(submission: pd.DataFrame) -> None:
    """
    Validate the submission format.

    The function verifies that the submission contains
    the expected columns and that predictions are valid.

    Args:
        submission:
            Submission DataFrame expected to contain
            ``ROW_ID`` and ``prediction``.

    Raises:
        ValueError:
            If the submission format is invalid.
    """

    required_columns = ["ROW_ID", "prediction"]

    missing_columns = [
        col for col in required_columns
        if col not in submission.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing submission columns: {missing_columns}")

    if submission["ROW_ID"].isnull().any():
        raise ValueError("ROW_ID contains missing values.")

    if submission["prediction"].isnull().any():
        raise ValueError("prediction contains missing values.")

    if not set(submission["prediction"].unique()).issubset({0, 1}):
        raise ValueError("Predictions must be binary values in {0, 1}.")


def export_submission_csv(
    submission: pd.DataFrame,
    filename: str = "submission_v001.csv"
) -> Path:
    """
    Export a submission file.

    The function saves the submission DataFrame
    to the submissions directory using the
    specified filename.

    Args:
        submission:
            Submission DataFrame.

        filename:
            Name of the output csv file.

    Returns:
        Path:
            Path to the exported submission file.
    """

    SUBMISSION_PATH.mkdir(parents=True, exist_ok=True)

    output_path = SUBMISSION_PATH / filename

    submission.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")

    return output_path


def main() -> None:
    """
    Train the final model and generate a submission file.

    The function performs the complete inference pipeline:

    - load the training and test datasets;
    - create the binary classification target;
    - train the logistic regression pipeline on the
    selected feature set;
    - generate predictions for the test set;
    - validate the submission format;
    - export the submission as a csv file.
    """

    X_train = load_X_train()
    X_test = load_X_test()
    y_train = load_y_train()

    df_train, _ = create_class_column(X_train, y_train)

    feature_cols = ret_features(df_train)

    model = build_logistic_regression_pipeline()

    model.fit(
        df_train[feature_cols],
        df_train["class"],
    )

    submission = X_test[["ROW_ID"]].copy()
    submission["prediction"] = model.predict(X_test[feature_cols])

    _validate_submission_format(submission)

    export_submission_csv(submission)


if __name__ == "__main__":
    main()