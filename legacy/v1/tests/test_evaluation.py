"""Tests for temporal model evaluation diagnostics."""

import math

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation import evaluate_model_on_folds


FEATURE_COLUMNS = ["signal"]
EXPECTED_DIAGNOSTIC_COLUMNS = {
    "n_train",
    "n_valid",
    "train_accuracy",
    "valid_accuracy",
    "train_roc_auc",
    "valid_roc_auc",
    "train_log_loss",
    "valid_log_loss",
    "accuracy_gap",
    "roc_auc_gap",
    "log_loss_gap",
    "fit_time_seconds",
    "train_prediction_time_seconds",
    "valid_prediction_time_seconds",
}


def _build_synthetic_dataframe() -> pd.DataFrame:
    rows = []
    row_id = 0

    for date in ["001", "002", "003", "004"]:
        for signal, target in [
            (-2.0, 0),
            (-1.0, 0),
            (1.0, 1),
            (2.0, 1),
        ]:
            rows.append(
                {
                    "ROW_ID": row_id,
                    "TS": date,
                    "signal": signal,
                    "class": target,
                }
            )
            row_id += 1

    return pd.DataFrame(rows)


def _build_two_folds() -> list[dict]:
    return [
        {
            "fold": 10,
            "train_start": "001",
            "train_end": "002",
            "valid_start": "003",
            "valid_end": "003",
        },
        {
            "fold": 42,
            "train_start": "001",
            "train_end": "003",
            "valid_start": "004",
            "valid_end": "004",
        },
    ]


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    max_iter=100,
                ),
            ),
        ]
    )


def test_evaluation_preserves_fold_ids_and_returns_two_rows() -> None:
    results = evaluate_model_on_folds(
        df=_build_synthetic_dataframe(),
        folds=_build_two_folds(),
        model=_build_pipeline(),
        feature_cols=FEATURE_COLUMNS,
    )

    assert results["fold"].tolist() == [10, 42]
    assert len(results) == 2


def test_evaluation_returns_diagnostics_and_preserves_legacy_metrics() -> None:
    results = evaluate_model_on_folds(
        df=_build_synthetic_dataframe(),
        folds=_build_two_folds(),
        model=_build_pipeline(),
        feature_cols=FEATURE_COLUMNS,
    )

    assert EXPECTED_DIAGNOSTIC_COLUMNS.issubset(results.columns)

    for _, row in results.iterrows():
        assert row["accuracy"] == pytest.approx(row["valid_accuracy"])
        assert row["roc_auc"] == pytest.approx(row["valid_roc_auc"])
        assert row["log_loss"] == pytest.approx(row["valid_log_loss"])

        assert row["accuracy_gap"] == pytest.approx(
            row["train_accuracy"] - row["valid_accuracy"]
        )
        assert row["roc_auc_gap"] == pytest.approx(
            row["train_roc_auc"] - row["valid_roc_auc"]
        )
        assert row["log_loss_gap"] == pytest.approx(
            row["valid_log_loss"] - row["train_log_loss"]
        )

        assert row["n_train"] > 0
        assert row["n_valid"] > 0
        assert row["n_valid"] == row["n_valid_predictions"]

        for timing_column in [
            "fit_time_seconds",
            "train_prediction_time_seconds",
            "valid_prediction_time_seconds",
        ]:
            assert math.isfinite(row[timing_column])
            assert row[timing_column] >= 0.0


def test_validation_aliases_and_gaps_use_distinct_validation_metrics() -> None:
    df = _build_synthetic_dataframe()
    df = df[df["TS"].isin(["001", "002", "003"])].copy()

    validation_mask = df["TS"] == "003"
    df.loc[validation_mask, "class"] = (
        1 - df.loc[validation_mask, "class"]
    )

    fold = {
        "fold": 7,
        "train_start": "001",
        "train_end": "002",
        "valid_start": "003",
        "valid_end": "003",
    }

    results = evaluate_model_on_folds(
        df=df,
        folds=[fold],
        model=_build_pipeline(),
        feature_cols=FEATURE_COLUMNS,
    )

    row = results.iloc[0]

    assert row["train_accuracy"] != row["valid_accuracy"]
    assert row["train_roc_auc"] != row["valid_roc_auc"]
    assert row["train_log_loss"] != row["valid_log_loss"]

    assert row["accuracy"] == pytest.approx(row["valid_accuracy"])
    assert row["accuracy"] != pytest.approx(row["train_accuracy"])

    assert row["roc_auc"] == pytest.approx(row["valid_roc_auc"])
    assert row["roc_auc"] != pytest.approx(row["train_roc_auc"])

    assert row["log_loss"] == pytest.approx(row["valid_log_loss"])
    assert row["log_loss"] != pytest.approx(row["train_log_loss"])

    assert row["accuracy_gap"] == pytest.approx(
        row["train_accuracy"] - row["valid_accuracy"]
    )
    assert row["roc_auc_gap"] == pytest.approx(
        row["train_roc_auc"] - row["valid_roc_auc"]
    )
    assert row["log_loss_gap"] == pytest.approx(
        row["valid_log_loss"] - row["train_log_loss"]
    )

    assert row["accuracy_gap"] != pytest.approx(0.0)
    assert row["roc_auc_gap"] != pytest.approx(0.0)
    assert row["log_loss_gap"] != pytest.approx(0.0)


def test_evaluation_leaves_original_pipeline_unfitted() -> None:
    pipeline = _build_pipeline()

    evaluate_model_on_folds(
        df=_build_synthetic_dataframe(),
        folds=_build_two_folds(),
        model=pipeline,
        feature_cols=FEATURE_COLUMNS,
    )

    assert not hasattr(pipeline.named_steps["imputer"], "statistics_")
    assert not hasattr(pipeline.named_steps["classifier"], "classes_")


def test_fold_without_identifier_raises_explicit_key_error() -> None:
    invalid_fold = _build_two_folds()[0].copy()
    invalid_fold.pop("fold")

    with pytest.raises(
        KeyError,
        match=r"Each fold must contain .*Missing keys: \['fold'\]",
    ):
        evaluate_model_on_folds(
            df=_build_synthetic_dataframe(),
            folds=[invalid_fold],
            model=_build_pipeline(),
            feature_cols=FEATURE_COLUMNS,
        )


def test_single_class_validation_has_no_roc_auc_or_gap() -> None:
    df = _build_synthetic_dataframe()
    df = df[df["TS"].isin(["001", "002", "003"])].copy()
    df.loc[df["TS"] == "003", "class"] = 1

    fold = {
        "fold": 7,
        "train_start": "001",
        "train_end": "002",
        "valid_start": "003",
        "valid_end": "003",
    }

    results = evaluate_model_on_folds(
        df=df,
        folds=[fold],
        model=_build_pipeline(),
        feature_cols=FEATURE_COLUMNS,
    )

    row = results.iloc[0]

    assert pd.isna(row["valid_roc_auc"])
    assert pd.isna(row["roc_auc"])
    assert pd.isna(row["roc_auc_gap"])
