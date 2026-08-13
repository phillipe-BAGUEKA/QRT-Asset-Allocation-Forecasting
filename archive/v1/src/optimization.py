"""Minimal Optuna infrastructure for Gradient Boosting optimization."""

from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd

from src.boosting_models import build_gradboosting_pipeline
from src.evaluation import evaluate_model_on_folds


REQUIRED_RESULT_COLUMNS = {
    "fold",
    "train_roc_auc",
    "valid_roc_auc",
    "roc_auc_gap",
    "valid_log_loss",
    "log_loss_gap",
    "fit_time_seconds",
}

METRIC_COLUMNS = [
    "train_roc_auc",
    "valid_roc_auc",
    "roc_auc_gap",
    "valid_log_loss",
    "log_loss_gap",
    "fit_time_seconds",
]


def suggest_gradient_boosting_params(
    trial: optuna.trial.Trial,
) -> dict[str, Any]:
    """Suggest the provisional Gradient Boosting search parameters."""
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.20,
            log=True,
        ),
        "n_estimators": trial.suggest_int(
            "n_estimators",
            50,
            300,
            step=25,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            1,
            4,
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            20,
            500,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0,
            step=0.1,
        ),
        "max_features": trial.suggest_categorical(
            "max_features",
            ["sqrt", "log2", None],
        ),
    }


def _validate_and_extract_metrics(
    results: pd.DataFrame,
    folds: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Validate fold coverage and return finite numerical metrics."""
    if not isinstance(results, pd.DataFrame):
        raise ValueError(
            "evaluate_model_on_folds must return a pandas DataFrame."
        )

    missing_columns = sorted(
        REQUIRED_RESULT_COLUMNS.difference(results.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required optimization columns: {missing_columns}."
        )

    if len(results) != len(folds):
        raise ValueError(
            "Evaluation results must contain exactly one row per fold. "
            f"Expected {len(folds)}, found {len(results)}."
        )

    expected_fold_ids = []

    for fold_index, fold in enumerate(folds, start=1):
        if "fold" not in fold:
            raise KeyError(
                f"Fold configuration {fold_index} is missing key 'fold'."
            )

        expected_fold_ids.append(fold["fold"])

    observed_fold_ids = results["fold"].tolist()

    if observed_fold_ids != expected_fold_ids:
        raise ValueError(
            "Evaluation fold identifiers or order do not match the "
            f"provided folds. Expected {expected_fold_ids}, found "
            f"{observed_fold_ids}."
        )

    metric_values: dict[str, np.ndarray] = {}

    for column in METRIC_COLUMNS:
        try:
            values = results[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Metric column '{column}' must be numerical."
            ) from error

        if not np.isfinite(values).all():
            raise ValueError(
                f"Metric column '{column}' contains missing or "
                "non-finite values."
            )

        metric_values[column] = values

    for column in ("train_roc_auc", "valid_roc_auc"):
        values = metric_values[column]

        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(
                f"Metric column '{column}' must lie in [0, 1]."
            )

    if (metric_values["valid_log_loss"] < 0.0).any():
        raise ValueError(
            "Metric column 'valid_log_loss' must be non-negative."
        )

    if (metric_values["fit_time_seconds"] < 0.0).any():
        raise ValueError(
            "Metric column 'fit_time_seconds' must be non-negative."
        )

    return metric_values


def create_gradient_boosting_objective(
    df: pd.DataFrame,
    folds: list[dict[str, Any]],
    feature_cols: list[str],
    target_col: str = "class",
    random_state: int = 42,
) -> Callable[[optuna.trial.Trial], float]:
    """Create an Optuna objective for temporal Gradient Boosting evaluation."""

    def objective(trial: optuna.trial.Trial) -> float:
        parameters = suggest_gradient_boosting_params(trial)
        model = build_gradboosting_pipeline(
            **parameters,
            random_state=random_state,
        )

        results = evaluate_model_on_folds(
            df=df,
            folds=folds,
            model=model,
            feature_cols=feature_cols,
            target_col=target_col,
        )

        metrics = _validate_and_extract_metrics(results, folds)
        valid_roc_auc = metrics["valid_roc_auc"]

        diagnostics = {
            "fold_valid_roc_auc": [
                float(value)
                for value in valid_roc_auc
            ],
            "std_valid_roc_auc": float(
                np.std(valid_roc_auc, ddof=0)
            ),
            "worst_valid_roc_auc": float(
                np.min(valid_roc_auc)
            ),
            "mean_train_roc_auc": float(
                np.mean(metrics["train_roc_auc"])
            ),
            "mean_roc_auc_gap": float(
                np.mean(metrics["roc_auc_gap"])
            ),
            "mean_valid_log_loss": float(
                np.mean(metrics["valid_log_loss"])
            ),
            "mean_log_loss_gap": float(
                np.mean(metrics["log_loss_gap"])
            ),
            "total_fit_time_seconds": float(
                np.sum(metrics["fit_time_seconds"])
            ),
        }

        for attribute_name, attribute_value in diagnostics.items():
            trial.set_user_attr(attribute_name, attribute_value)

        return float(np.mean(valid_roc_auc))

    return objective


def create_gradient_boosting_study(
    seed: int = 42,
    study_name: str | None = None,
    storage: Any = None,
    load_if_exists: bool = False,
) -> optuna.study.Study:
    """Create an in-memory-by-default maximizing Optuna study."""
    sampler = optuna.samplers.TPESampler(seed=seed)

    return optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )
