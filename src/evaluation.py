from typing import Callable, Any

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


REQUIRED_FOLD_KEYS = ["train_start", "train_end", "valid_start", "valid_end"]
BINARY_CLASSES = {0, 1}


def _validate_folds(folds: list[dict]) -> None:
    """Validate temporal fold definitions."""
    if not folds:
        raise ValueError("Array of folds is empty.")

    for fold in folds:
        if not all(key in fold for key in REQUIRED_FOLD_KEYS):
            raise KeyError(f"Each fold must contain: {REQUIRED_FOLD_KEYS}.")


def _validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Validate that all required columns are present in a DataFrame."""
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def _get_temporal_split(
    df: pd.DataFrame,
    fold: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronological train and validation subsets for one fold."""
    fold_train = df[
        (df["TS"] >= fold["train_start"])
        & (df["TS"] <= fold["train_end"])
    ]

    fold_valid = df[
        (df["TS"] >= fold["valid_start"])
        & (df["TS"] <= fold["valid_end"])
    ]

    if fold_train.empty or fold_valid.empty:
        raise ValueError("Training or validation DataFrame is empty.")

    return fold_train, fold_valid


def _validate_prediction_dataframe(
    fold_valid: pd.DataFrame,
    validation_id_before: pd.Series,
    validation_length_before: int,
    context: str
) -> None:
    """Validate prediction output structure and binary values."""
    if not isinstance(fold_valid, pd.DataFrame):
        raise ValueError(f"{context} must preserve a pandas DataFrame.")

    if "prediction" not in fold_valid.columns:
        raise ValueError(f"{context} must return a DataFrame with a 'prediction' column.")

    if (
        not validation_id_before.equals(fold_valid["ROW_ID"])
        or validation_length_before != len(fold_valid)
    ):
        raise ValueError(
            f"{context} changed the number or order of validation observations."
        )

    if fold_valid["prediction"].isnull().any():
        raise ValueError("Predictions contain missing values.")

    if not set(fold_valid["prediction"].unique()).issubset(BINARY_CLASSES):
        raise ValueError("Predictions must be binary values in {0, 1}.")


def evaluate_baseline_on_folds(
    df: pd.DataFrame,
    folds: list[dict],
    baseline_function: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """
    Evaluate a deterministic prediction baseline across temporal folds.

    For each fold, the function creates chronological training and
    validation subsets, applies the selected baseline, validates the
    returned predictions and computes fold-level accuracy.

    Args:
        df:
            Complete labeled training dataset.

            The DataFrame must contain:

            - ``TS``: temporal identifier;
            - ``class``: binary target;
            - ``ROW_ID``: unique observation identifier.

        folds:
            List of temporal fold dictionaries.

            Each dictionary must contain:

            - ``train_start``;
            - ``train_end``;
            - ``valid_start``;
            - ``valid_end``.

        baseline_function:
            Callable baseline receiving the current training and
            validation DataFrames.

            The callable must return the validation DataFrame, in the
            same order, with an additional binary column named
            ``prediction``.

    Returns:
        pd.DataFrame:
            One row per fold containing temporal boundaries,
            class rates, accuracy and prediction counts.

    Raises:
        ValueError:
            If folds are empty, required columns are missing, train or
            validation subsets are empty, predictions are missing,
            invalid, null, or validation observations are modified.

        KeyError:
            If a fold definition is incomplete.

    Notes:
        This function is dedicated to deterministic baseline strategies.
        Learned models should be evaluated with ``evaluate_model_on_folds``.
    """
    _validate_folds(folds)
    _validate_required_columns(df, ["TS", "class", "ROW_ID"])

    results = []

    for fold_index, fold in enumerate(folds, start=1):
        fold_train, fold_valid = _get_temporal_split(df, fold)

        fold_train_positive_rate = (fold_train["class"] == 1).mean()
        fold_valid_positive_rate = (fold_valid["class"] == 1).mean()

        fold_valid = fold_valid.copy()
        validation_length_before = len(fold_valid)
        validation_id_before = fold_valid["ROW_ID"].copy()

        fold_valid = baseline_function(fold_train, fold_valid)

        _validate_prediction_dataframe(
            fold_valid=fold_valid,
            validation_id_before=validation_id_before,
            validation_length_before=validation_length_before,
            context="Baseline evaluation"
        )

        fold_accuracy = accuracy_score(fold_valid["class"], fold_valid["prediction"])
        fold_n_correct_predictions = (
            fold_valid["class"] == fold_valid["prediction"]
        ).sum()

        results.append({
            "fold": fold_index,
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "valid_start": fold["valid_start"],
            "valid_end": fold["valid_end"],
            "train_positive_rate": fold_train_positive_rate,
            "valid_positive_rate": fold_valid_positive_rate,
            "accuracy": fold_accuracy,
            "n_valid_predictions": len(fold_valid),
            "n_correct_predictions": fold_n_correct_predictions,
        })

    return pd.DataFrame(results)


def evaluate_model_on_folds(
    df: pd.DataFrame,
    folds: list[dict],
    model: Any,
    feature_cols: list[str],
    target_col: str = "class",
) -> pd.DataFrame:
    """
    Evaluate a supervised classification model across temporal folds.

    For each fold, the function creates chronological training and
    validation subsets, clones the input model, fits it on the training
    subset, generates validation predictions and predicted probabilities,
    then computes fold-level evaluation metrics.

    Args:
        df:
            Complete labeled dataset.

            The DataFrame must contain:

            - ``TS``: temporal identifier;
            - ``ROW_ID``: unique observation identifier;
            - the target column;
            - all feature columns.

        folds:
            List of temporal fold dictionaries.

            Each dictionary must contain:

            - ``train_start``;
            - ``train_end``;
            - ``valid_start``;
            - ``valid_end``.

        model:
            Untrained Scikit-Learn compatible estimator or Pipeline.

            The model must implement:

            - ``fit``;
            - ``predict``;
            - ``predict_proba``.

        feature_cols:
            List of feature columns used for training and validation
            inference.

        target_col:
            Name of the binary target column. Default is ``"class"``.

    Returns:
        pd.DataFrame:
            One row per fold containing temporal boundaries,
            class rates, accuracy, log-loss, ROC-AUC when computable,
            and prediction counts.

    Raises:
        ValueError:
            If folds are empty, required columns are missing, feature
            columns are absent, train or validation subsets are empty,
            predictions are missing, invalid, null, or validation
            observations are modified.

        KeyError:
            If a fold definition is incomplete.

    Notes:
        The model is cloned before every fold to ensure that preprocessing
        statistics and learned parameters are never shared across
        temporal folds.
    """
    _validate_folds(folds)
    _validate_required_columns(df, ["TS", target_col, "ROW_ID"])
    _validate_required_columns(df, feature_cols)

    results = []

    for fold_index, fold in enumerate(folds, start=1):
        fold_model = clone(model)
        fold_train, fold_valid = _get_temporal_split(df, fold)

        X_train = fold_train[feature_cols]
        y_train = fold_train[target_col]

        X_valid = fold_valid[feature_cols]
        y_valid = fold_valid[target_col]

        fold_train_positive_rate = (y_train == 1).mean()
        fold_valid_positive_rate = (y_valid == 1).mean()

        fold_valid = fold_valid.copy()
        validation_length_before = len(fold_valid)
        validation_id_before = fold_valid["ROW_ID"].copy()

        fold_model.fit(X_train, y_train)

        fold_valid["prediction"] = fold_model.predict(X_valid)
        fold_valid["predicted_proba"] = fold_model.predict_proba(X_valid)[:, 1]

        _validate_prediction_dataframe(
            fold_valid=fold_valid,
            validation_id_before=validation_id_before,
            validation_length_before=validation_length_before,
            context="Model evaluation"
        )

        fold_accuracy = accuracy_score(y_valid, fold_valid["prediction"])
        fold_log_loss = log_loss(y_valid, fold_valid["predicted_proba"])

        if y_valid.nunique() == 2:
            fold_roc_auc = roc_auc_score(y_valid, fold_valid["predicted_proba"])
        else:
            fold_roc_auc = None

        fold_n_correct_predictions = (
            y_valid.values == fold_valid["prediction"].values
        ).sum()

        results.append({
            "fold": fold_index,
            "train_start": fold["train_start"],
            "train_end": fold["train_end"],
            "valid_start": fold["valid_start"],
            "valid_end": fold["valid_end"],
            "train_positive_rate": fold_train_positive_rate,
            "valid_positive_rate": fold_valid_positive_rate,
            "accuracy": fold_accuracy,
            "log_loss": fold_log_loss,
            "roc_auc": fold_roc_auc,
            "n_valid_predictions": len(fold_valid),
            "n_correct_predictions": fold_n_correct_predictions,
        })

    return pd.DataFrame(results)