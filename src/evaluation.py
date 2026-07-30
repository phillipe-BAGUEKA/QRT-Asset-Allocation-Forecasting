"""
Evaluation utilities for the QRT Asset Allocation Forecasting project.

This module contains reusable functions for:

- deterministic baseline evaluation;
- supervised model evaluation across temporal folds;
- fold-level model comparison;
- generation of out-of-fold prediction tables for model diagnosis.

All learned models are cloned before each fold so that fitted parameters
and preprocessing statistics are never shared between temporal windows.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


REQUIRED_FOLD_KEYS = [
    "fold",
    "train_start",
    "train_end",
    "valid_start",
    "valid_end",
]

BINARY_CLASSES = {0, 1}

MANDATORY_DIAGNOSTIC_COLUMNS = [
    "ROW_ID",
    "TS",
    "fold",
    "model",
    "y_true",
    "y_pred",
    "y_proba",
]

OPTIONAL_DIAGNOSTIC_COLUMNS = [
    "GROUP",
    "ALLOCATION",
]


def _validate_models(models: list[dict[str, Any]]) -> None:
    """
    Validate model configurations used for diagnostic training.

    Each configuration must contain:

    - ``name``: a unique, non-empty model name;
    - ``estimator``: an unfitted Scikit-Learn compatible estimator or pipeline
      implementing ``fit``, ``predict`` and ``predict_proba``.

    Args:
        models:
            List of model configuration dictionaries.

    Raises:
        ValueError:
            If the list is empty, a configuration is malformed, a name is
            missing or duplicated, or an estimator does not expose the
            required methods.
    """
    if not models:
        raise ValueError("The model configuration list is empty.")

    model_names: list[str] = []

    for index, model_config in enumerate(models, start=1):
        if not isinstance(model_config, dict):
            raise ValueError(
                f"Model configuration {index} must be a dictionary."
            )

        missing_keys = {
            key
            for key in ("name", "estimator")
            if key not in model_config
        }

        if missing_keys:
            raise ValueError(
                f"Model configuration {index} is missing keys: "
                f"{sorted(missing_keys)}."
            )

        model_name = model_config["name"]
        estimator = model_config["estimator"]

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError(
                f"Model configuration {index} has an invalid name."
            )

        required_methods = ("fit", "predict", "predict_proba")
        missing_methods = [
            method
            for method in required_methods
            if not callable(getattr(estimator, method, None))
        ]

        if missing_methods:
            raise ValueError(
                f"Model '{model_name}' does not implement: "
                f"{missing_methods}."
            )

        model_names.append(model_name)

    duplicated_names = pd.Index(model_names)[
        pd.Index(model_names).duplicated()
    ].tolist()

    if duplicated_names:
        raise ValueError(
            f"Model names must be unique. Duplicates: {duplicated_names}."
        )


def _validate_folds(folds: list[dict]) -> None:
    """
    Validate temporal fold definitions.

    Args:
        folds:
            List of temporal fold dictionaries.

    Raises:
        ValueError:
            If the list is empty or fold identifiers are duplicated.

        KeyError:
            If a fold does not contain every required key.
    """
    if not folds:
        raise ValueError("The fold list is empty.")

    fold_ids = []

    for fold in folds:
        missing_keys = [
            key
            for key in REQUIRED_FOLD_KEYS
            if key not in fold
        ]

        if missing_keys:
            raise KeyError(
                f"Each fold must contain {REQUIRED_FOLD_KEYS}. "
                f"Missing keys: {missing_keys}."
            )

        fold_ids.append(fold["fold"])

    duplicated_fold_ids = pd.Index(fold_ids)[
        pd.Index(fold_ids).duplicated()
    ].tolist()

    if duplicated_fold_ids:
        raise ValueError(
            f"Fold identifiers must be unique. "
            f"Duplicates: {duplicated_fold_ids}."
        )


def _validate_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
) -> None:
    """
    Validate that all required columns are present in a DataFrame.

    Args:
        df:
            DataFrame to validate.

        required_cols:
            Columns that must be present.

    Raises:
        ValueError:
            If the DataFrame is invalid or required columns are missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input dataset must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The input DataFrame is empty.")

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}.")


def _validate_feature_columns(feature_cols: list[str]) -> None:
    """
    Validate the feature-column list.

    Raises:
        ValueError:
            If no feature is supplied or feature names are duplicated.
    """
    if not feature_cols:
        raise ValueError("The feature column list is empty.")

    duplicated_features = pd.Index(feature_cols)[
        pd.Index(feature_cols).duplicated()
    ].tolist()

    if duplicated_features:
        raise ValueError(
            f"Feature names must be unique. Duplicates: "
            f"{duplicated_features}."
        )


def _validate_binary_target(
    target: pd.Series,
    context: str,
    require_both_classes: bool = False,
) -> None:
    """
    Validate a binary target series.

    Args:
        target:
            Target values to validate.

        context:
            Text describing the validation context.

        require_both_classes:
            Whether both binary classes must be present.

    Raises:
        ValueError:
            If values are missing, non-binary, or only one class is present
            when two classes are required.
    """
    if target.isna().any():
        raise ValueError(f"{context} contains missing target values.")

    unique_values = set(target.unique())

    if not unique_values.issubset(BINARY_CLASSES):
        raise ValueError(
            f"{context} must contain only binary values in {{0, 1}}. "
            f"Found: {sorted(unique_values)}."
        )

    if require_both_classes and unique_values != BINARY_CLASSES:
        raise ValueError(
            f"{context} must contain both target classes 0 and 1."
        )


def _get_temporal_split(
    df: pd.DataFrame,
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create chronological train and validation subsets for one fold.

    Raises:
        ValueError:
            If a subset is empty or train and validation overlap in time.
    """
    fold_train = df[
        (df["TS"] >= fold["train_start"])
        & (df["TS"] <= fold["train_end"])
    ]

    fold_valid = df[
        (df["TS"] >= fold["valid_start"])
        & (df["TS"] <= fold["valid_end"])
    ]

    if fold_train.empty or fold_valid.empty:
        raise ValueError(
            f"Training or validation data is empty for fold "
            f"{fold['fold']}."
        )

    if fold_train["TS"].max() >= fold_valid["TS"].min():
        raise ValueError(
            f"Temporal leakage detected in fold {fold['fold']}: "
            "training must strictly precede validation."
        )

    return fold_train, fold_valid


def _validate_prediction_dataframe(
    fold_valid: pd.DataFrame,
    validation_id_before: pd.Series,
    validation_length_before: int,
    context: str,
) -> None:
    """
    Validate prediction output structure and binary values.
    """
    if not isinstance(fold_valid, pd.DataFrame):
        raise ValueError(
            f"{context} must preserve a pandas DataFrame."
        )

    if "prediction" not in fold_valid.columns:
        raise ValueError(
            f"{context} must return a DataFrame containing "
            "'prediction'."
        )

    if (
        validation_length_before != len(fold_valid)
        or not validation_id_before.reset_index(drop=True).equals(
            fold_valid["ROW_ID"].reset_index(drop=True)
        )
    ):
        raise ValueError(
            f"{context} changed the number or order of validation "
            "observations."
        )

    if fold_valid["prediction"].isna().any():
        raise ValueError(f"{context} predictions contain missing values.")

    prediction_values = set(fold_valid["prediction"].unique())

    if not prediction_values.issubset(BINARY_CLASSES):
        raise ValueError(
            f"{context} predictions must be binary values in {{0, 1}}. "
            f"Found: {sorted(prediction_values)}."
        )


def diagnostic_dataframe(
    df: pd.DataFrame,
    target_col: str = "class",
) -> pd.DataFrame:
    """
    Build and validate a standardized diagnostic prediction DataFrame.

    The target, predicted class and predicted probability are renamed to
    ``y_true``, ``y_pred`` and ``y_proba``. ``GROUP`` and ``ALLOCATION``
    are preserved when available but are not mandatory.

    Args:
        df:
            Validation DataFrame containing identifiers, model metadata,
            target values and model predictions.

        target_col:
            Name of the source target column.

    Returns:
        A standardized diagnostic DataFrame.

    Raises:
        ValueError:
            If required columns are missing, target values are invalid,
            probabilities are missing, non-finite or outside [0, 1].
    """
    df = df.copy()

    required_source_columns = [
        "ROW_ID",
        "TS",
        "fold",
        "model",
        target_col,
        "prediction",
        "predicted_proba",
    ]

    _validate_required_columns(df, required_source_columns)
    _validate_binary_target(
        target=df[target_col],
        context="Diagnostic target",
    )

    probabilities = df["predicted_proba"]

    invalid_probability_mask = (
        probabilities.isna()
        | ~np.isfinite(probabilities)
        | probabilities.lt(0.0)
        | probabilities.gt(1.0)
    )

    if invalid_probability_mask.any():
        invalid_count = int(invalid_probability_mask.sum())
        raise ValueError(
            f"Found {invalid_count} invalid predicted probabilities. "
            "Probabilities must be finite and lie in [0, 1]."
        )

    output_columns = required_source_columns.copy()
    output_columns.extend(
        column
        for column in OPTIONAL_DIAGNOSTIC_COLUMNS
        if column in df.columns
    )

    diagnostic_df = df[output_columns].rename(
        columns={
            target_col: "y_true",
            "prediction": "y_pred",
            "predicted_proba": "y_proba",
        }
    )

    return diagnostic_df


def _validate_complete_diagnostic_dataframe(
    diagnostic_df: pd.DataFrame,
    expected_model_names: list[str],
) -> None:
    """
    Validate the final out-of-fold diagnostic table.

    Checks:

    - mandatory output columns;
    - absence of duplicated model-observation pairs;
    - presence of every expected model;
    - equal prediction counts across models.
    """
    _validate_required_columns(
        diagnostic_df,
        MANDATORY_DIAGNOSTIC_COLUMNS,
    )

    duplicated_pairs = diagnostic_df.duplicated(
        subset=["model", "ROW_ID"],
        keep=False,
    )

    if duplicated_pairs.any():
        duplicate_count = int(duplicated_pairs.sum())
        raise ValueError(
            f"Found {duplicate_count} duplicated (model, ROW_ID) rows. "
            "Validation folds may overlap or predictions may have been "
            "added more than once."
        )

    observed_model_names = set(diagnostic_df["model"].unique())
    expected_model_name_set = set(expected_model_names)

    if observed_model_names != expected_model_name_set:
        missing_models = sorted(
            expected_model_name_set - observed_model_names
        )
        unexpected_models = sorted(
            observed_model_names - expected_model_name_set
        )

        raise ValueError(
            "Unexpected model coverage in diagnostic results. "
            f"Missing: {missing_models}; unexpected: {unexpected_models}."
        )

    prediction_counts = diagnostic_df.groupby(
        "model",
        observed=True,
    ).size()

    if prediction_counts.nunique() != 1:
        raise ValueError(
            "Models do not have the same number of out-of-fold "
            f"predictions: {prediction_counts.to_dict()}."
        )


def evaluate_baseline_on_folds(
    df: pd.DataFrame,
    folds: list[dict],
    baseline_function: Callable[
        [pd.DataFrame, pd.DataFrame],
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Evaluate a deterministic prediction baseline across temporal folds.

    For each fold, the function creates chronological training and
    validation subsets, applies the selected baseline, validates the
    returned predictions and computes fold-level accuracy.
    """
    _validate_folds(folds)
    _validate_required_columns(df, ["TS", "class", "ROW_ID"])

    results = []

    for fold_index, fold in enumerate(folds, start=1):
        fold_train, fold_valid = _get_temporal_split(df, fold)

        fold_train_positive_rate = (
            fold_train["class"] == 1
        ).mean()
        fold_valid_positive_rate = (
            fold_valid["class"] == 1
        ).mean()

        fold_valid = fold_valid.copy()
        validation_length_before = len(fold_valid)
        validation_id_before = fold_valid["ROW_ID"].copy()

        fold_valid = baseline_function(fold_train, fold_valid)

        _validate_prediction_dataframe(
            fold_valid=fold_valid,
            validation_id_before=validation_id_before,
            validation_length_before=validation_length_before,
            context="Baseline evaluation",
        )

        fold_accuracy = accuracy_score(
            fold_valid["class"],
            fold_valid["prediction"],
        )

        fold_n_correct_predictions = (
            fold_valid["class"] == fold_valid["prediction"]
        ).sum()

        results.append(
            {
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
            }
        )

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

    The model is cloned before every fold to prevent fitted parameters
    and preprocessing statistics from being shared across validation
    windows.
    """
    _validate_folds(folds)
    _validate_feature_columns(feature_cols)
    _validate_required_columns(
        df,
        ["TS", target_col, "ROW_ID", *feature_cols],
    )
    _validate_binary_target(
        df[target_col],
        context="Complete target",
        require_both_classes=True,
    )

    results = []

    for fold_index, fold in enumerate(folds, start=1):
        fold_model = clone(model)
        fold_train, fold_valid = _get_temporal_split(df, fold)

        X_train = fold_train[feature_cols]
        y_train = fold_train[target_col]

        X_valid = fold_valid[feature_cols]
        y_valid = fold_valid[target_col]

        _validate_binary_target(
            y_train,
            context=f"Training target for fold {fold['fold']}",
            require_both_classes=True,
        )
        _validate_binary_target(
            y_valid,
            context=f"Validation target for fold {fold['fold']}",
        )

        fold_train_positive_rate = (y_train == 1).mean()
        fold_valid_positive_rate = (y_valid == 1).mean()

        fold_valid = fold_valid.copy()
        validation_length_before = len(fold_valid)
        validation_id_before = fold_valid["ROW_ID"].copy()

        fold_model.fit(X_train, y_train)

        fold_valid["prediction"] = fold_model.predict(X_valid)
        fold_valid["predicted_proba"] = (
            fold_model.predict_proba(X_valid)[:, 1]
        )

        _validate_prediction_dataframe(
            fold_valid=fold_valid,
            validation_id_before=validation_id_before,
            validation_length_before=validation_length_before,
            context=f"Model evaluation, fold {fold['fold']}",
        )

        fold_accuracy = accuracy_score(
            y_valid,
            fold_valid["prediction"],
        )
        fold_log_loss = log_loss(
            y_valid,
            fold_valid["predicted_proba"],
            labels=[0, 1],
        )

        if y_valid.nunique() == 2:
            fold_roc_auc = roc_auc_score(
                y_valid,
                fold_valid["predicted_proba"],
            )
        else:
            fold_roc_auc = None

        fold_n_correct_predictions = (
            y_valid.to_numpy()
            == fold_valid["prediction"].to_numpy()
        ).sum()

        results.append(
            {
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
            }
        )

    return pd.DataFrame(results)


def compare_model_results(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
) -> pd.DataFrame:
    """
    Compare two model-evaluation tables fold by fold.

    Deltas are calculated as ``model_b - model_a``.

    Therefore:

    - positive ``delta_accuracy`` is favourable to model B;
    - positive ``delta_roc_auc`` is favourable to model B;
    - negative ``delta_log_loss`` is favourable to model B.
    """
    required_columns = {
        "fold",
        "accuracy",
        "log_loss",
        "roc_auc",
    }

    missing_a = required_columns - set(results_a.columns)
    missing_b = required_columns - set(results_b.columns)

    if missing_a:
        raise ValueError(
            f"Missing columns in results_a: {sorted(missing_a)}."
        )

    if missing_b:
        raise ValueError(
            f"Missing columns in results_b: {sorted(missing_b)}."
        )

    results_a_renamed = results_a[
        ["fold", "accuracy", "log_loss", "roc_auc"]
    ].rename(
        columns={
            "accuracy": f"accuracy_{model_a_name}",
            "log_loss": f"log_loss_{model_a_name}",
            "roc_auc": f"roc_auc_{model_a_name}",
        }
    )

    results_b_renamed = results_b[
        ["fold", "accuracy", "log_loss", "roc_auc"]
    ].rename(
        columns={
            "accuracy": f"accuracy_{model_b_name}",
            "log_loss": f"log_loss_{model_b_name}",
            "roc_auc": f"roc_auc_{model_b_name}",
        }
    )

    comparison = results_a_renamed.merge(
        results_b_renamed,
        on="fold",
        how="inner",
        validate="one_to_one",
    )

    comparison["delta_accuracy"] = (
        comparison[f"accuracy_{model_b_name}"]
        - comparison[f"accuracy_{model_a_name}"]
    )

    comparison["delta_log_loss"] = (
        comparison[f"log_loss_{model_b_name}"]
        - comparison[f"log_loss_{model_a_name}"]
    )

    comparison["delta_roc_auc"] = (
        comparison[f"roc_auc_{model_b_name}"]
        - comparison[f"roc_auc_{model_a_name}"]
    )

    return comparison


def diagnostic_training(
    df: pd.DataFrame,
    folds: list[dict],
    models: list[dict[str, Any]],
    feature_cols: list[str],
    target_col: str = "class",
) -> pd.DataFrame:
    """
    Generate out-of-fold predictions for multiple classification models.

    Every model is cloned and trained independently on each temporal fold.
    Only validation predictions are retained. The resulting table is the
    common data source for probability, calibration, date, group,
    allocation and intra-date ranking diagnostics.

    Args:
        df:
            Complete labelled training dataset.

        folds:
            Expanding-window temporal fold definitions.

        models:
            Model configurations with unique ``name`` and ``estimator``
            entries.

        feature_cols:
            Features used by every compared model.

        target_col:
            Binary target column. Default is ``"class"``.

    Returns:
        A DataFrame where each row represents one out-of-fold prediction
        made by one model for one validation observation.

    Raises:
        ValueError:
            If inputs, estimators, targets, probabilities or final
            prediction coverage are invalid.
    """
    _validate_folds(folds)
    _validate_models(models)
    _validate_feature_columns(feature_cols)
    _validate_required_columns(
        df,
        ["TS", target_col, "ROW_ID", *feature_cols],
    )
    _validate_binary_target(
        df[target_col],
        context="Complete target",
        require_both_classes=True,
    )

    diagnostic_frames: list[pd.DataFrame] = []

    for model_config in models:
        model_name = model_config["name"]
        model_estimator = model_config["estimator"]

        for fold in folds:
            fold_model = clone(model_estimator)
            fold_train, fold_valid = _get_temporal_split(df, fold)

            X_train = fold_train[feature_cols]
            y_train = fold_train[target_col]

            X_valid = fold_valid[feature_cols]
            y_valid = fold_valid[target_col]

            _validate_binary_target(
                y_train,
                context=(
                    f"Training target for model '{model_name}', "
                    f"fold {fold['fold']}"
                ),
                require_both_classes=True,
            )
            _validate_binary_target(
                y_valid,
                context=(
                    f"Validation target for model '{model_name}', "
                    f"fold {fold['fold']}"
                ),
            )

            fold_valid = fold_valid.copy()
            validation_length_before = len(fold_valid)
            validation_id_before = fold_valid["ROW_ID"].copy()

            fold_model.fit(X_train, y_train)

            fold_valid["prediction"] = fold_model.predict(X_valid)
            fold_valid["predicted_proba"] = (
                fold_model.predict_proba(X_valid)[:, 1]
            )
            fold_valid["model"] = model_name
            fold_valid["fold"] = fold["fold"]

            _validate_prediction_dataframe(
                fold_valid=fold_valid,
                validation_id_before=validation_id_before,
                validation_length_before=validation_length_before,
                context=(
                    f"Diagnostic predictions for model "
                    f"'{model_name}', fold {fold['fold']}"
                ),
            )

            fold_diagnostic = diagnostic_dataframe(
                fold_valid,
                target_col=target_col,
            )

            diagnostic_frames.append(fold_diagnostic)

    if not diagnostic_frames:
        raise ValueError("No diagnostic prediction was generated.")

    diagnostic_df = pd.concat(
        diagnostic_frames,
        ignore_index=True,
    )

    expected_model_names = [
        model_config["name"]
        for model_config in models
    ]

    _validate_complete_diagnostic_dataframe(
        diagnostic_df=diagnostic_df,
        expected_model_names=expected_model_names,
    )

    return diagnostic_df