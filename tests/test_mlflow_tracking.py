"""Tests for local MLflow tracking around Optuna studies and trials."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import mlflow
import numpy as np
import optuna
import pytest
from mlflow import MlflowClient
from optuna.trial import TrialState

from src.mlflow_tracking import (
    configure_local_mlflow,
    create_mlflow_tracked_objective,
    run_optuna_study_with_mlflow,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_ROOT_PATHS = [
    PROJECT_ROOT / "mlflow_data",
    PROJECT_ROOT / "mlflow.db",
    PROJECT_ROOT / "mlruns",
    PROJECT_ROOT / "mlartifacts",
]

TRIAL_USER_ATTRIBUTES = {
    "fold_valid_roc_auc": [np.float64(0.61), np.float64(0.59)],
    "std_valid_roc_auc": np.float64(0.01),
    "worst_valid_roc_auc": np.float64(0.59),
    "mean_train_roc_auc": np.float64(0.72),
    "mean_roc_auc_gap": np.float64(0.12),
    "mean_valid_log_loss": np.float64(0.68),
    "mean_log_loss_gap": np.float64(0.04),
    "total_fit_time_seconds": np.float64(1.25),
}


class ExpectedTrialFailure(RuntimeError):
    """Exception used to validate failed MLflow child runs."""


def _end_all_active_runs() -> None:
    while mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture(autouse=True)
def isolate_mlflow_global_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original_tracking_uri = mlflow.get_tracking_uri()
    _end_all_active_runs()
    monkeypatch.chdir(tmp_path)

    yield

    _end_all_active_runs()
    mlflow.set_tracking_uri(original_tracking_uri)


def _configure_test_tracking(
    tmp_path: Path,
) -> tuple[str, MlflowClient, Path, Path]:
    database_path = tmp_path / "tracking" / "mlflow.db"
    artifact_root = tmp_path / "artifacts" / "root"
    experiment_name = f"test-{uuid4()}"
    experiment_id = configure_local_mlflow(
        tracking_db_path=database_path,
        artifact_root=artifact_root,
        experiment_name=experiment_name,
    )
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

    return experiment_id, client, database_path, artifact_root


def _set_trial_user_attributes(trial: optuna.trial.BaseTrial) -> None:
    for key, value in TRIAL_USER_ATTRIBUTES.items():
        trial.set_user_attr(key, value)


def _successful_objective(
    trial: optuna.trial.BaseTrial,
    calls: list[int] | None = None,
) -> float:
    if calls is not None:
        calls.append(trial.number)

    trial.suggest_float("learning_rate", 0.01, 0.20)
    _set_trial_user_attributes(trial)
    return 0.60


def _child_runs(
    client: MlflowClient,
    experiment_id: str,
    parent_run_id: str,
) -> list[Any]:
    return [
        run
        for run in client.search_runs([experiment_id])
        if run.data.tags.get("mlflow.parentRunId") == parent_run_id
    ]


@contextmanager
def _running_parent(
    client: MlflowClient,
    experiment_id: str,
    run_name: str,
):
    parent = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
    )

    try:
        yield parent
    except BaseException:
        client.set_terminated(parent.info.run_id, status="FAILED")
        raise
    else:
        client.set_terminated(parent.info.run_id, status="FINISHED")


def _load_json_artifact(
    client: MlflowClient,
    run_id: str,
    artifact_path: str,
    destination: Path,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    local_path = client.download_artifacts(
        run_id=run_id,
        path=artifact_path,
        dst_path=str(destination),
    )

    return json.loads(Path(local_path).read_text(encoding="utf-8"))


def test_configure_local_mlflow_creates_and_reuses_experiment(
    tmp_path: Path,
) -> None:
    root_state_before = {
        path: path.exists()
        for path in MLFLOW_ROOT_PATHS
    }
    database_path = tmp_path / "backend" / "nested" / "mlflow.db"
    artifact_root = tmp_path / "artifacts" / "nested"
    experiment_name = f"configuration-{uuid4()}"

    experiment_id = configure_local_mlflow(
        database_path,
        artifact_root,
        experiment_name,
    )
    reused_experiment_id = configure_local_mlflow(
        database_path,
        artifact_root,
        experiment_name,
    )

    expected_tracking_uri = (
        f"sqlite:///{database_path.resolve().as_posix()}"
    )
    client = MlflowClient(tracking_uri=expected_tracking_uri)
    experiment = client.get_experiment(experiment_id)

    assert experiment_id == reused_experiment_id
    assert mlflow.get_tracking_uri() == expected_tracking_uri
    assert database_path.parent.is_dir()
    assert database_path.is_file()
    assert artifact_root.is_dir()
    assert experiment.artifact_location.rstrip("/") == (
        artifact_root.resolve().as_uri().rstrip("/")
    )
    assert {
        path: path.exists()
        for path in MLFLOW_ROOT_PATHS
    } == root_state_before


def test_configure_local_mlflow_rejects_artifact_location_change(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "mlflow.db"
    experiment_name = f"artifact-mismatch-{uuid4()}"

    configure_local_mlflow(
        database_path,
        tmp_path / "artifacts-a",
        experiment_name,
    )

    with pytest.raises(
        ValueError,
        match="already uses artifact_location",
    ):
        configure_local_mlflow(
            database_path,
            tmp_path / "artifacts-b",
            experiment_name,
        )


def test_tracked_objective_logs_successful_child_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id, client, _, _ = _configure_test_tracking(tmp_path)
    objective_calls: list[int] = []

    def forbidden_autolog(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("mlflow.autolog must not be called")

    monkeypatch.setattr(mlflow, "autolog", forbidden_autolog)

    with _running_parent(
        client,
        experiment_id,
        "manual-parent",
    ) as parent_run:
        tracked_objective = create_mlflow_tracked_objective(
            objective=lambda trial: _successful_objective(
                trial,
                objective_calls,
            ),
            experiment_id=experiment_id,
            parent_run_id=parent_run.info.run_id,
            fixed_params={"random_state": np.int64(42)},
            tags={
                "model_family": "gradient_boosting",
                "feature_set": "RET_1_to_RET_20",
            },
        )
        trial = optuna.trial.FixedTrial(
            {"learning_rate": 0.05},
            number=7,
        )

        score = tracked_objective(trial)
        parent_run_id = parent_run.info.run_id

    children = _child_runs(client, experiment_id, parent_run_id)

    assert score == 0.60
    assert objective_calls == [7]
    assert len(children) == 1

    child = children[0]
    parent_from_api = mlflow.get_parent_run(child.info.run_id)

    assert parent_from_api is not None
    assert parent_from_api.info.run_id == parent_run_id
    assert child.info.status == "FINISHED"
    assert child.data.params == {
        "learning_rate": "0.05",
        "random_state": "42",
    }
    assert child.data.metrics == {
        "mean_valid_roc_auc": pytest.approx(0.60),
        "std_valid_roc_auc": pytest.approx(0.01),
        "worst_valid_roc_auc": pytest.approx(0.59),
        "mean_train_roc_auc": pytest.approx(0.72),
        "mean_roc_auc_gap": pytest.approx(0.12),
        "mean_valid_log_loss": pytest.approx(0.68),
        "mean_log_loss_gap": pytest.approx(0.04),
        "total_fit_time_seconds": pytest.approx(1.25),
    }
    assert child.data.tags["optuna_trial_number"] == "7"
    assert child.data.tags["model_family"] == "gradient_boosting"
    assert child.data.tags["optimization_method"] == "optuna_tpe"
    assert child.data.tags["optuna_trial_state"] == "COMPLETE"
    assert child.data.tags["feature_set"] == "RET_1_to_RET_20"

    artifact_names = {
        artifact.path
        for artifact in client.list_artifacts(child.info.run_id)
    }
    summary = _load_json_artifact(
        client,
        child.info.run_id,
        "trial_summary.json",
        tmp_path / "download-success",
    )

    assert artifact_names == {"trial_summary.json"}
    assert summary["trial_number"] == 7
    assert summary["parameters"] == {"learning_rate": 0.05}
    assert summary["objective_value"] == pytest.approx(0.60)
    assert summary["fixed_params"] == {"random_state": 42}
    assert summary["user_attrs"]["fold_valid_roc_auc"] == [0.61, 0.59]
    assert summary["tags"]["optuna_trial_state"] == "COMPLETE"


def test_tracked_objective_logs_failure_and_reraises_original(
    tmp_path: Path,
) -> None:
    experiment_id, client, _, _ = _configure_test_tracking(tmp_path)
    original_error = ExpectedTrialFailure("failure " + "x" * 2_000)

    def failing_objective(trial: optuna.trial.BaseTrial) -> float:
        trial.suggest_float("learning_rate", 0.01, 0.20)
        raise original_error

    with _running_parent(
        client,
        experiment_id,
        "failure-parent",
    ) as parent_run:
        tracked_objective = create_mlflow_tracked_objective(
            objective=failing_objective,
            experiment_id=experiment_id,
            parent_run_id=parent_run.info.run_id,
            fixed_params={"random_state": 42},
        )

        with pytest.raises(ExpectedTrialFailure) as error_info:
            tracked_objective(
                optuna.trial.FixedTrial(
                    {"learning_rate": 0.05},
                    number=3,
                )
            )

        parent_run_id = parent_run.info.run_id

    assert error_info.value is original_error

    children = _child_runs(client, experiment_id, parent_run_id)
    assert len(children) == 1

    child = children[0]
    assert child.info.status == "FAILED"
    assert child.data.params == {
        "learning_rate": "0.05",
        "random_state": "42",
    }
    assert child.data.tags["optuna_trial_state"] == "FAILED"
    assert child.data.tags["error_type"] == "ExpectedTrialFailure"
    assert len(child.data.tags["error_message"]) == 1_000

    summary = _load_json_artifact(
        client,
        child.info.run_id,
        "trial_summary.json",
        tmp_path / "download-failure",
    )
    assert summary["objective_value"] is None
    assert summary["tags"]["optuna_trial_state"] == "FAILED"


def test_failure_logging_error_does_not_mask_objective_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment_id, client, _, _ = _configure_test_tracking(tmp_path)
    original_error = ExpectedTrialFailure("original failure")

    def failing_objective(trial: optuna.trial.BaseTrial) -> float:
        trial.suggest_float("learning_rate", 0.01, 0.20)
        raise original_error

    def failing_log_params(parameters: dict[str, Any]) -> None:
        raise RuntimeError("tracking failure")

    monkeypatch.setattr(mlflow, "log_params", failing_log_params)

    with _running_parent(
        client,
        experiment_id,
        "logging-failure-parent",
    ) as parent_run:
        tracked_objective = create_mlflow_tracked_objective(
            objective=failing_objective,
            experiment_id=experiment_id,
            parent_run_id=parent_run.info.run_id,
        )

        with pytest.raises(ExpectedTrialFailure) as error_info:
            tracked_objective(
                optuna.trial.FixedTrial(
                    {"learning_rate": 0.05},
                    number=4,
                )
            )

        parent_run_id = parent_run.info.run_id

    assert error_info.value is original_error
    assert any(
        "MLflow failure logging also failed" in note
        for note in getattr(original_error, "__notes__", [])
    )
    children = _child_runs(client, experiment_id, parent_run_id)
    assert len(children) == 1
    assert children[0].info.status == "FAILED"
    assert children[0].data.tags["optuna_trial_state"] == "FAILED"
    assert children[0].data.tags["error_type"] == (
        "ExpectedTrialFailure"
    )


def test_parameter_collisions_allow_equal_and_reject_different_values(
    tmp_path: Path,
) -> None:
    experiment_id, client, _, _ = _configure_test_tracking(tmp_path)

    with _running_parent(
        client,
        experiment_id,
        "collision-parent",
    ) as parent_run:
        parent_run_id = parent_run.info.run_id

        equal_wrapper = create_mlflow_tracked_objective(
            objective=_successful_objective,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
            fixed_params={"learning_rate": 0.05},
        )
        equal_wrapper(
            optuna.trial.FixedTrial(
                {"learning_rate": 0.05},
                number=1,
            )
        )

        conflicting_wrapper = create_mlflow_tracked_objective(
            objective=_successful_objective,
            experiment_id=experiment_id,
            parent_run_id=parent_run_id,
            fixed_params={"learning_rate": 0.10},
        )

        with pytest.raises(ValueError, match="Parameter collision"):
            conflicting_wrapper(
                optuna.trial.FixedTrial(
                    {"learning_rate": 0.05},
                    number=2,
                )
            )

    children = _child_runs(client, experiment_id, parent_run_id)
    runs_by_name = {
        run.data.tags["mlflow.runName"]: run
        for run in children
    }

    assert runs_by_name["trial_1"].data.params == {
        "learning_rate": "0.05"
    }
    assert runs_by_name["trial_1"].info.status == "FINISHED"
    assert runs_by_name["trial_2"].info.status == "FAILED"


@pytest.mark.parametrize(
    "tags",
    [
        {"model_family": "xgboost"},
        {"optuna_trial_state": "COMPLETE"},
        {"error_message": "caller value"},
    ],
)
def test_child_reserved_tags_cannot_be_overridden(
    tags: dict[str, str],
) -> None:
    objective_calls: list[int] = []
    wrapper = create_mlflow_tracked_objective(
        objective=lambda trial: _successful_objective(
            trial,
            objective_calls,
        ),
        experiment_id="unused",
        parent_run_id="unused",
        tags=tags,
    )

    with pytest.raises(ValueError, match="reserved"):
        wrapper(optuna.trial.FixedTrial({}, number=0))

    assert objective_calls == []


def test_study_run_logs_session_counts_and_global_best_trial(
    tmp_path: Path,
) -> None:
    experiment_id, client, _, _ = _configure_test_tracking(tmp_path)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": 0.90},
            distributions={
                "x": optuna.distributions.FloatDistribution(0.0, 1.0)
            },
            value=0.95,
            state=TrialState.COMPLETE,
            user_attrs={"origin": "historical"},
        )
    )

    def session_objective(trial: optuna.trial.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)

        if trial.number == 1:
            raise ExpectedTrialFailure("caught session failure")

        _set_trial_user_attributes(trial)
        return 0.50

    returned_study, parent_run_id = run_optuna_study_with_mlflow(
        study=study,
        objective=session_objective,
        experiment_id=experiment_id,
        parent_run_name="study-with-history",
        n_trials=2,
        timeout=30.0,
        parent_params={
            "n_folds": 4,
            "sampler_seed": 42,
            "random_state": 42,
        },
        parent_tags={
            "feature_set": "RET_1_to_RET_20",
            "project_stage": "tracking_infrastructure",
            "git_branch": "cleanup/pre-main",
        },
        fixed_trial_params={"random_state": 42},
        child_tags={"feature_set": "RET_1_to_RET_20"},
        catch=(ExpectedTrialFailure,),
    )

    assert returned_study is study

    parent = client.get_run(parent_run_id)
    assert parent.info.status == "FINISHED"
    assert parent.data.params == {
        "requested_n_trials": "2",
        "timeout_seconds": "30.0",
        "objective_metric": "mean_valid_roc_auc",
        "sampler_class": "RandomSampler",
        "pruner_class": "NopPruner",
        "n_folds": "4",
        "sampler_seed": "42",
        "random_state": "42",
    }
    assert parent.data.metrics == {
        "session_number_of_trials": pytest.approx(2.0),
        "study_total_number_of_trials": pytest.approx(3.0),
        "session_complete_trials": pytest.approx(1.0),
        "session_failed_trials": pytest.approx(1.0),
        "session_pruned_trials": pytest.approx(0.0),
        "study_best_trial_number": pytest.approx(0.0),
        "study_best_objective_value": pytest.approx(0.95),
    }
    assert parent.data.tags["project"] == "QRT_Asset_Allocation"
    assert parent.data.tags["model_family"] == "gradient_boosting"
    assert (
        parent.data.tags["validation_protocol"]
        == "expanding_window_4_folds"
    )
    assert parent.data.tags["feature_set"] == "RET_1_to_RET_20"

    children = _child_runs(client, experiment_id, parent_run_id)
    assert len(children) == 2
    assert sorted(child.info.status for child in children) == [
        "FAILED",
        "FINISHED",
    ]

    best_trial_summary = _load_json_artifact(
        client,
        parent_run_id,
        "best_trial.json",
        tmp_path / "download-best",
    )
    assert best_trial_summary == {
        "scope": "global_study_best_trial",
        "best_trial_number": 0,
        "best_value": pytest.approx(0.95),
        "best_params": {"x": pytest.approx(0.90)},
        "best_user_attrs": {"origin": "historical"},
    }


def test_enqueued_waiting_trial_is_counted_in_current_session(
    tmp_path: Path,
) -> None:
    experiment_id, client, _, _ = _configure_test_tracking(tmp_path)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=42),
        pruner=optuna.pruners.NopPruner(),
    )
    study.enqueue_trial({"learning_rate": 0.05})

    assert len(study.trials) == 1
    assert study.trials[0].state == TrialState.WAITING

    returned_study, parent_run_id = run_optuna_study_with_mlflow(
        study=study,
        objective=_successful_objective,
        experiment_id=experiment_id,
        parent_run_name="study-with-enqueued-trial",
        n_trials=1,
    )

    assert returned_study is study
    assert len(study.trials) == 1
    assert study.trials[0].state == TrialState.COMPLETE

    parent = client.get_run(parent_run_id)
    assert parent.data.metrics == {
        "session_number_of_trials": pytest.approx(1.0),
        "study_total_number_of_trials": pytest.approx(1.0),
        "session_complete_trials": pytest.approx(1.0),
        "session_failed_trials": pytest.approx(0.0),
        "session_pruned_trials": pytest.approx(0.0),
        "study_best_trial_number": pytest.approx(0.0),
        "study_best_objective_value": pytest.approx(0.60),
    }

    children = _child_runs(client, experiment_id, parent_run_id)
    assert len(children) == 1


def test_study_parent_run_fails_when_optimize_raises(
    tmp_path: Path,
) -> None:
    experiment_id, client, _, _ = _configure_test_tracking(tmp_path)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.NopPruner(),
    )
    parent_run_name = f"uncaught-parent-{uuid4()}"
    original_error = ExpectedTrialFailure("uncaught failure")

    def failing_objective(trial: optuna.trial.Trial) -> float:
        raise original_error

    with pytest.raises(ExpectedTrialFailure) as error_info:
        run_optuna_study_with_mlflow(
            study=study,
            objective=failing_objective,
            experiment_id=experiment_id,
            parent_run_name=parent_run_name,
            n_trials=1,
        )

    assert error_info.value is original_error

    matching_parents = [
        run
        for run in client.search_runs([experiment_id])
        if run.data.tags.get("mlflow.runName") == parent_run_name
    ]
    assert len(matching_parents) == 1
    assert matching_parents[0].info.status == "FAILED"

    children = _child_runs(
        client,
        experiment_id,
        matching_parents[0].info.run_id,
    )
    assert len(children) == 1
    assert children[0].info.status == "FAILED"


@pytest.mark.parametrize(
    ("n_trials", "timeout", "expected_message"),
    [
        (0, None, "n_trials"),
        (-1, None, "n_trials"),
        (True, None, "n_trials"),
        (1, 0.0, "timeout"),
        (1, -1.0, "timeout"),
        (1, np.inf, "timeout"),
    ],
)
def test_study_run_validates_trial_count_and_timeout(
    n_trials: Any,
    timeout: Any,
    expected_message: str,
) -> None:
    study = optuna.create_study(direction="maximize")

    with pytest.raises(ValueError, match=expected_message):
        run_optuna_study_with_mlflow(
            study=study,
            objective=lambda trial: 0.5,
            experiment_id="unused",
            parent_run_name="unused",
            n_trials=n_trials,
            timeout=timeout,
        )

    assert mlflow.active_run() is None


def test_parent_reserved_tag_cannot_be_overridden() -> None:
    study = optuna.create_study(direction="maximize")

    with pytest.raises(ValueError, match="reserved"):
        run_optuna_study_with_mlflow(
            study=study,
            objective=lambda trial: 0.5,
            experiment_id="unused",
            parent_run_name="unused",
            n_trials=1,
            parent_tags={"project": "another_project"},
        )

    assert mlflow.active_run() is None
