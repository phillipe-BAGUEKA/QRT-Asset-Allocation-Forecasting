"""Local MLflow tracking utilities for Optuna studies and trials."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
from mlflow import MlflowClient
from optuna.trial import TrialState


PathInput = str | os.PathLike[str]

CHILD_METRIC_USER_ATTRIBUTES = (
    "std_valid_roc_auc",
    "worst_valid_roc_auc",
    "mean_train_roc_auc",
    "mean_roc_auc_gap",
    "mean_valid_log_loss",
    "mean_log_loss_gap",
    "total_fit_time_seconds",
)

RESERVED_TAG_KEYS = {
    "project",
    "model_family",
    "optimization_method",
    "validation_protocol",
    "optuna_trial_number",
    "optuna_trial_state",
    "error_type",
    "error_message",
}

ERROR_MESSAGE_MAX_LENGTH = 1_000


def _to_json_serializable(value: Any) -> Any:
    """Convert nested values to JSON-serializable Python objects."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, np.generic):
        return _to_json_serializable(value.item())

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _to_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_to_json_serializable(item) for item in value]

    return str(value)


def _to_mlflow_param(value: Any) -> Any:
    """Convert a value to an MLflow-compatible parameter value."""
    converted = _to_json_serializable(value)

    if converted is None:
        return "None"

    if isinstance(converted, (str, bool, int, float)):
        return converted

    return json.dumps(converted, sort_keys=True)


def _same_value(left: Any, right: Any) -> bool:
    """Compare values after conversion to stable JSON-compatible forms."""
    return _to_json_serializable(left) == _to_json_serializable(right)


def _merge_parameters(
    primary: Mapping[str, Any] | None,
    secondary: Mapping[str, Any] | None,
    *,
    primary_name: str,
    secondary_name: str,
) -> dict[str, Any]:
    """Merge parameter mappings while rejecting conflicting values."""
    merged = dict(primary or {})

    for key, value in dict(secondary or {}).items():
        if key in merged and not _same_value(merged[key], value):
            raise ValueError(
                f"Parameter collision for '{key}' between "
                f"{primary_name} and {secondary_name}."
            )

        if key not in merged:
            merged[key] = value

    return merged


def _merge_tags(
    standard_tags: Mapping[str, Any],
    provided_tags: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge tags without allowing reserved-tag overrides."""
    merged = dict(standard_tags)

    for key, value in dict(provided_tags or {}).items():
        if key in RESERVED_TAG_KEYS:
            if key not in standard_tags:
                raise ValueError(
                    f"Tag '{key}' is reserved and cannot be supplied here."
                )

            if not _same_value(standard_tags[key], value):
                raise ValueError(
                    f"Tag '{key}' is reserved and cannot be overridden."
                )

        if key not in merged:
            merged[key] = value

    return {
        key: str(_to_mlflow_param(value))
        for key, value in merged.items()
    }


def _log_parameters(parameters: Mapping[str, Any]) -> None:
    """Log a parameter mapping after MLflow-compatible conversion."""
    if parameters:
        mlflow.log_params(
            {
                key: _to_mlflow_param(value)
                for key, value in parameters.items()
            }
        )


def _log_client_parameters(
    client: MlflowClient,
    run_id: str,
    parameters: Mapping[str, Any],
) -> None:
    """Log parameters to a run that is not on MLflow's active stack."""
    for key, value in parameters.items():
        client.log_param(run_id, key, _to_mlflow_param(value))


def _log_client_metrics(
    client: MlflowClient,
    run_id: str,
    metrics: Mapping[str, float],
) -> None:
    """Log metrics to a run that is not on MLflow's active stack."""
    for key, value in metrics.items():
        client.log_metric(run_id, key, value)


def _finite_metric(value: Any, metric_name: str) -> float:
    """Convert one metric to a finite Python float."""
    try:
        metric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"MLflow metric '{metric_name}' must be numerical."
        ) from error

    if not math.isfinite(metric_value):
        raise ValueError(
            f"MLflow metric '{metric_name}' must be finite."
        )

    return metric_value


def _build_trial_summary(
    *,
    trial: optuna.trial.Trial,
    objective_value: Any,
    fixed_params: Mapping[str, Any],
    effective_tags: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the JSON artifact shared by successful and failed trials."""
    return _to_json_serializable(
        {
            "trial_number": int(trial.number),
            "parameters": dict(trial.params),
            "objective_value": objective_value,
            "user_attrs": dict(trial.user_attrs),
            "fixed_params": dict(fixed_params),
            "tags": dict(effective_tags),
        }
    )


def configure_local_mlflow(
    tracking_db_path: PathInput,
    artifact_root: PathInput,
    experiment_name: str,
) -> str:
    """Configure SQLite tracking and create or reuse one local experiment."""
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("experiment_name must be a non-empty string.")

    database_path = Path(tracking_db_path).expanduser().resolve()
    resolved_artifact_root = Path(artifact_root).expanduser().resolve()

    database_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_artifact_root.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{database_path.as_posix()}"
    artifact_location = resolved_artifact_root.as_uri().rstrip("/")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
        return str(experiment_id)

    existing_artifact_location = (
        experiment.artifact_location or ""
    ).rstrip("/")

    if existing_artifact_location != artifact_location:
        raise ValueError(
            f"Experiment '{experiment_name}' already uses artifact_location "
            f"'{existing_artifact_location}', not '{artifact_location}'."
        )

    return str(experiment.experiment_id)


def create_mlflow_tracked_objective(
    objective: Callable[[optuna.trial.Trial], float],
    experiment_id: str,
    parent_run_id: str,
    fixed_params: Mapping[str, Any] | None = None,
    tags: Mapping[str, Any] | None = None,
    run_name_prefix: str = "trial",
) -> Callable[[optuna.trial.Trial], float]:
    """Wrap an Optuna objective in one explicitly parented MLflow run."""
    if not isinstance(run_name_prefix, str) or not run_name_prefix.strip():
        raise ValueError("run_name_prefix must be a non-empty string.")

    fixed_parameters = dict(fixed_params or {})

    def tracked_objective(trial: optuna.trial.Trial) -> float:
        standard_tags = {
            "optuna_trial_number": str(trial.number),
            "model_family": "gradient_boosting",
            "optimization_method": "optuna_tpe",
        }
        effective_tags = _merge_tags(standard_tags, tags)
        run_name = f"{run_name_prefix}_{trial.number}"

        with mlflow.start_run(
            experiment_id=str(experiment_id),
            run_name=run_name,
            parent_run_id=parent_run_id,
            tags=effective_tags,
        ):
            try:
                score = objective(trial)
                merged_parameters = _merge_parameters(
                    trial.params,
                    fixed_parameters,
                    primary_name="trial.params",
                    secondary_name="fixed_params",
                )
                _log_parameters(merged_parameters)

                metrics = {
                    "mean_valid_roc_auc": _finite_metric(
                        score,
                        "mean_valid_roc_auc",
                    )
                }

                for metric_name in CHILD_METRIC_USER_ATTRIBUTES:
                    if metric_name in trial.user_attrs:
                        metrics[metric_name] = _finite_metric(
                            trial.user_attrs[metric_name],
                            metric_name,
                        )

                mlflow.log_metrics(metrics)
                mlflow.set_tag("optuna_trial_state", "COMPLETE")

                complete_tags = {
                    **effective_tags,
                    "optuna_trial_state": "COMPLETE",
                }
                mlflow.log_dict(
                    _build_trial_summary(
                        trial=trial,
                        objective_value=score,
                        fixed_params=fixed_parameters,
                        effective_tags=complete_tags,
                    ),
                    "trial_summary.json",
                )

                return score

            except Exception as original_error:
                try:
                    failure_tags = {
                        **effective_tags,
                        "optuna_trial_state": "FAILED",
                        "error_type": type(original_error).__name__,
                        "error_message": str(original_error)[
                            :ERROR_MESSAGE_MAX_LENGTH
                        ],
                    }
                    mlflow.set_tags(failure_tags)

                    merged_parameters = _merge_parameters(
                        trial.params,
                        fixed_parameters,
                        primary_name="trial.params",
                        secondary_name="fixed_params",
                    )
                    _log_parameters(merged_parameters)
                    mlflow.log_dict(
                        _build_trial_summary(
                            trial=trial,
                            objective_value=None,
                            fixed_params=fixed_parameters,
                            effective_tags=failure_tags,
                        ),
                        "trial_summary.json",
                    )
                except Exception as logging_error:
                    original_error.add_note(
                        "MLflow failure logging also failed: "
                        f"{type(logging_error).__name__}: {logging_error}"
                    )

                raise

    return tracked_objective


def _validate_study_run_arguments(
    n_trials: int,
    timeout: float | None,
) -> None:
    """Validate the bounded study-run arguments."""
    if isinstance(n_trials, bool) or not isinstance(n_trials, int):
        raise ValueError("n_trials must be a strictly positive integer.")

    if n_trials <= 0:
        raise ValueError("n_trials must be a strictly positive integer.")

    if timeout is not None:
        if isinstance(timeout, bool):
            raise ValueError("timeout must be strictly positive when set.")

        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "timeout must be strictly positive when set."
            ) from error

        if not math.isfinite(timeout_value) or timeout_value <= 0.0:
            raise ValueError("timeout must be strictly positive when set.")


def run_optuna_study_with_mlflow(
    study: optuna.study.Study,
    objective: Callable[[optuna.trial.Trial], float],
    experiment_id: str,
    parent_run_name: str,
    n_trials: int,
    timeout: float | None = None,
    parent_params: Mapping[str, Any] | None = None,
    parent_tags: Mapping[str, Any] | None = None,
    fixed_trial_params: Mapping[str, Any] | None = None,
    child_tags: Mapping[str, Any] | None = None,
    catch: tuple[type[Exception], ...] = (),
) -> tuple[optuna.study.Study, str]:
    """Run one Optuna study session inside a parent MLflow run."""
    _validate_study_run_arguments(n_trials, timeout)

    trial_states_before = {
        trial.number: trial.state
        for trial in study.trials
    }

    standard_parent_params = {
        "requested_n_trials": n_trials,
        "timeout_seconds": timeout if timeout is not None else "None",
        "objective_metric": "mean_valid_roc_auc",
        "sampler_class": type(study.sampler).__name__,
        "pruner_class": type(study.pruner).__name__,
    }
    effective_parent_params = _merge_parameters(
        standard_parent_params,
        parent_params,
        primary_name="standard parent parameters",
        secondary_name="parent_params",
    )

    standard_parent_tags = {
        "project": "QRT_Asset_Allocation",
        "model_family": "gradient_boosting",
        "validation_protocol": "expanding_window_4_folds",
    }
    effective_parent_tags = _merge_tags(
        standard_parent_tags,
        parent_tags,
    )

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    parent_run = client.create_run(
        experiment_id=str(experiment_id),
        run_name=parent_run_name,
        tags=effective_parent_tags,
    )
    parent_run_id = parent_run.info.run_id

    try:
        _log_client_parameters(
            client,
            parent_run_id,
            effective_parent_params,
        )

        tracked_objective = create_mlflow_tracked_objective(
            objective=objective,
            experiment_id=str(experiment_id),
            parent_run_id=parent_run_id,
            fixed_params=fixed_trial_params,
            tags=child_tags,
        )

        study.optimize(
            tracked_objective,
            n_trials=n_trials,
            timeout=timeout,
            catch=catch,
        )

        all_trials = list(study.trials)
        # An enqueued trial already has a number before optimize. Count it in
        # this session when optimize consumes it and changes its WAITING state.
        # Trials that were historically terminal remain excluded.
        session_trials = [
            trial
            for trial in all_trials
            if (
                trial.number not in trial_states_before
                or (
                    trial_states_before[trial.number]
                    == TrialState.WAITING
                    and trial.state != TrialState.WAITING
                )
            )
        ]

        session_complete_trials = [
            trial
            for trial in session_trials
            if trial.state == TrialState.COMPLETE
        ]
        session_failed_trials = [
            trial
            for trial in session_trials
            if trial.state == TrialState.FAIL
        ]
        session_pruned_trials = [
            trial
            for trial in session_trials
            if trial.state == TrialState.PRUNED
        ]

        parent_metrics = {
            "session_number_of_trials": float(len(session_trials)),
            "study_total_number_of_trials": float(len(all_trials)),
            "session_complete_trials": float(
                len(session_complete_trials)
            ),
            "session_failed_trials": float(len(session_failed_trials)),
            "session_pruned_trials": float(len(session_pruned_trials)),
        }

        global_complete_trials = [
            trial
            for trial in all_trials
            if trial.state == TrialState.COMPLETE
        ]

        if global_complete_trials:
            best_trial = study.best_trial
            parent_metrics["study_best_trial_number"] = float(
                best_trial.number
            )
            parent_metrics["study_best_objective_value"] = float(
                best_trial.value
            )

            client.log_dict(
                parent_run_id,
                _to_json_serializable(
                    {
                        "scope": "global_study_best_trial",
                        "best_trial_number": int(best_trial.number),
                        "best_value": float(best_trial.value),
                        "best_params": dict(best_trial.params),
                        "best_user_attrs": dict(best_trial.user_attrs),
                    }
                ),
                "best_trial.json",
            )

        _log_client_metrics(client, parent_run_id, parent_metrics)
    except BaseException as original_error:
        try:
            client.set_terminated(parent_run_id, status="FAILED")
        except Exception as tracking_error:
            original_error.add_note(
                "MLflow parent failure finalization also failed: "
                f"{type(tracking_error).__name__}: {tracking_error}"
            )
        raise
    else:
        client.set_terminated(parent_run_id, status="FINISHED")

    return study, parent_run_id
