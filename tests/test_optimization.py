"""Unit tests for the minimal Optuna optimization infrastructure."""

from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

import src.optimization as optimization


FIXED_PARAMETERS = {
    "learning_rate": 0.05,
    "n_estimators": 100,
    "max_depth": 2,
    "min_samples_leaf": 50,
    "subsample": 0.8,
    "max_features": "sqrt",
}

EXPECTED_USER_ATTRIBUTES = {
    "fold_valid_roc_auc",
    "std_valid_roc_auc",
    "worst_valid_roc_auc",
    "mean_train_roc_auc",
    "mean_roc_auc_gap",
    "mean_valid_log_loss",
    "mean_log_loss_gap",
    "total_fit_time_seconds",
}


def _build_folds() -> list[dict[str, Any]]:
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


def _build_evaluation_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold": [10, 42],
            "train_roc_auc": [0.80, 0.70],
            "valid_roc_auc": [0.60, 0.40],
            "roc_auc_gap": [0.20, 0.30],
            "valid_log_loss": [0.65, 0.75],
            "log_loss_gap": [0.05, 0.10],
            "fit_time_seconds": [1.20, 2.30],
        }
    )


def _build_objective_with_result(
    monkeypatch: pytest.MonkeyPatch,
    results: pd.DataFrame,
) -> tuple[Any, optuna.trial.FixedTrial]:
    monkeypatch.setattr(
        optimization,
        "build_gradboosting_pipeline",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        optimization,
        "evaluate_model_on_folds",
        lambda **kwargs: results,
    )

    objective = optimization.create_gradient_boosting_objective(
        df=pd.DataFrame({"signal": [0.0]}),
        folds=_build_folds(),
        feature_cols=["signal"],
    )

    return objective, optuna.trial.FixedTrial(FIXED_PARAMETERS)


def test_suggest_gradient_boosting_params_returns_exact_search_values() -> None:
    trial = optuna.trial.FixedTrial(FIXED_PARAMETERS)

    parameters = optimization.suggest_gradient_boosting_params(trial)

    assert parameters == FIXED_PARAMETERS
    assert set(parameters) == {
        "learning_rate",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "max_features",
    }

    assert trial.distributions["learning_rate"] == (
        optuna.distributions.FloatDistribution(
            low=0.01,
            high=0.20,
            log=True,
        )
    )
    assert trial.distributions["n_estimators"] == (
        optuna.distributions.IntDistribution(
            low=50,
            high=300,
            step=25,
        )
    )
    assert trial.distributions["max_depth"] == (
        optuna.distributions.IntDistribution(
            low=1,
            high=4,
        )
    )
    assert trial.distributions["min_samples_leaf"] == (
        optuna.distributions.IntDistribution(
            low=20,
            high=500,
            log=True,
        )
    )
    assert trial.distributions["subsample"] == (
        optuna.distributions.FloatDistribution(
            low=0.6,
            high=1.0,
            step=0.1,
        )
    )
    assert trial.distributions["max_features"] == (
        optuna.distributions.CategoricalDistribution(
            choices=("sqrt", "log2", None),
        )
    )


def test_objective_transmits_parameters_and_evaluates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder_calls = []
    evaluation_calls = []
    model = object()
    df = pd.DataFrame({"signal": [0.0]})
    folds = _build_folds()

    def fake_builder(**kwargs: Any) -> object:
        builder_calls.append(kwargs)
        return model

    def fake_evaluation(**kwargs: Any) -> pd.DataFrame:
        evaluation_calls.append(kwargs)
        return _build_evaluation_results()

    monkeypatch.setattr(
        optimization,
        "build_gradboosting_pipeline",
        fake_builder,
    )
    monkeypatch.setattr(
        optimization,
        "evaluate_model_on_folds",
        fake_evaluation,
    )

    objective = optimization.create_gradient_boosting_objective(
        df=df,
        folds=folds,
        feature_cols=["signal"],
        target_col="label",
        random_state=123,
    )

    objective(optuna.trial.FixedTrial(FIXED_PARAMETERS))

    assert builder_calls == [
        {
            **FIXED_PARAMETERS,
            "random_state": 123,
        }
    ]
    assert len(evaluation_calls) == 1
    assert evaluation_calls[0] == {
        "df": df,
        "folds": folds,
        "model": model,
        "feature_cols": ["signal"],
        "target_col": "label",
    }


def test_objective_returns_score_and_exact_python_user_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective, trial = _build_objective_with_result(
        monkeypatch,
        _build_evaluation_results(),
    )

    score = objective(trial)

    assert type(score) is float
    assert score == pytest.approx(0.50)
    assert set(trial.user_attrs) == EXPECTED_USER_ATTRIBUTES
    assert trial.user_attrs["fold_valid_roc_auc"] == [0.60, 0.40]
    assert trial.user_attrs["std_valid_roc_auc"] == pytest.approx(0.10)
    assert trial.user_attrs["worst_valid_roc_auc"] == pytest.approx(0.40)
    assert trial.user_attrs["mean_train_roc_auc"] == pytest.approx(0.75)
    assert trial.user_attrs["mean_roc_auc_gap"] == pytest.approx(0.25)
    assert trial.user_attrs["mean_valid_log_loss"] == pytest.approx(0.70)
    assert trial.user_attrs["mean_log_loss_gap"] == pytest.approx(0.075)
    assert trial.user_attrs["total_fit_time_seconds"] == pytest.approx(3.50)

    for attribute_value in trial.user_attrs.values():
        if isinstance(attribute_value, list):
            assert all(type(value) is float for value in attribute_value)
        else:
            assert type(attribute_value) is float


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf])
def test_objective_rejects_non_finite_validation_roc_auc(
    monkeypatch: pytest.MonkeyPatch,
    invalid_value: float,
) -> None:
    results = _build_evaluation_results()
    results.loc[0, "valid_roc_auc"] = invalid_value
    objective, trial = _build_objective_with_result(monkeypatch, results)

    with pytest.raises(
        ValueError,
        match="valid_roc_auc.*missing or non-finite",
    ):
        objective(trial)


@pytest.mark.parametrize(
    ("column", "invalid_value", "expected_message"),
    [
        ("train_roc_auc", 1.01, "train_roc_auc.*\\[0, 1\\]"),
        ("valid_roc_auc", -0.01, "valid_roc_auc.*\\[0, 1\\]"),
        ("valid_log_loss", -0.01, "valid_log_loss.*non-negative"),
        ("fit_time_seconds", -0.01, "fit_time_seconds.*non-negative"),
    ],
)
def test_objective_rejects_metrics_outside_their_domains(
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    invalid_value: float,
    expected_message: str,
) -> None:
    results = _build_evaluation_results()
    results.loc[0, column] = invalid_value
    objective, trial = _build_objective_with_result(monkeypatch, results)

    with pytest.raises(ValueError, match=expected_message):
        objective(trial)


@pytest.mark.parametrize("column", ["roc_auc_gap", "log_loss_gap"])
def test_objective_rejects_non_finite_gap_metrics(
    monkeypatch: pytest.MonkeyPatch,
    column: str,
) -> None:
    results = _build_evaluation_results()
    results.loc[0, column] = np.nan
    objective, trial = _build_objective_with_result(monkeypatch, results)

    with pytest.raises(
        ValueError,
        match=f"{column}.*missing or non-finite",
    ):
        objective(trial)


def test_objective_rejects_incorrect_result_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _build_evaluation_results().iloc[[0]].copy()
    objective, trial = _build_objective_with_result(monkeypatch, results)

    with pytest.raises(ValueError, match="exactly one row per fold"):
        objective(trial)


def test_objective_rejects_incorrect_fold_identifiers_or_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _build_evaluation_results().iloc[::-1].reset_index(drop=True)
    objective, trial = _build_objective_with_result(monkeypatch, results)

    with pytest.raises(
        ValueError,
        match="fold identifiers or order do not match",
    ):
        objective(trial)


def test_objective_rejects_missing_required_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _build_evaluation_results().drop(columns="roc_auc_gap")
    objective, trial = _build_objective_with_result(monkeypatch, results)

    with pytest.raises(
        ValueError,
        match="Missing required optimization columns.*roc_auc_gap",
    ):
        objective(trial)


def test_study_factory_creates_empty_maximizing_in_memory_study() -> None:
    study = optimization.create_gradient_boosting_study(seed=123)

    assert study.direction == optuna.study.StudyDirection.MAXIMIZE
    assert isinstance(study.sampler, optuna.samplers.TPESampler)
    assert isinstance(study.pruner, optuna.pruners.NopPruner)
    assert isinstance(study._storage, optuna.storages.InMemoryStorage)
    assert study.trials == []


def test_study_factory_passes_configuration_with_nop_pruner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = object()
    study = object()
    sampler_seeds = []
    create_study_calls = []

    def fake_sampler(*, seed: int) -> object:
        sampler_seeds.append(seed)
        return sampler

    def fake_create_study(**kwargs: Any) -> object:
        create_study_calls.append(kwargs)
        return study

    monkeypatch.setattr(
        optimization.optuna.samplers,
        "TPESampler",
        fake_sampler,
    )
    monkeypatch.setattr(
        optimization.optuna,
        "create_study",
        fake_create_study,
    )

    returned_study = optimization.create_gradient_boosting_study(
        seed=123,
        study_name="technical-test",
        storage=None,
        load_if_exists=True,
    )

    assert returned_study is study
    assert sampler_seeds == [123]
    assert len(create_study_calls) == 1

    create_study_call = create_study_calls[0]

    assert set(create_study_call) == {
        "direction",
        "sampler",
        "pruner",
        "study_name",
        "storage",
        "load_if_exists",
    }
    assert create_study_call["direction"] == "maximize"
    assert create_study_call["sampler"] is sampler
    assert create_study_call["study_name"] == "technical-test"
    assert create_study_call["storage"] is None
    assert create_study_call["load_if_exists"] is True
    assert isinstance(
        create_study_call["pruner"],
        optuna.pruners.NopPruner,
    )
