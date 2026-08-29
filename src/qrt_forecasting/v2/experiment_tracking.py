'''Local MLflow tracking primitives for V2 boosting experiments.'''

from __future__ import annotations

import json
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import mlflow
import numpy as np
from mlflow.entities import RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

from qrt_forecasting.v2.boosting import EnginePilotResult


_CONFIGURED_EXPERIMENT_ID: str | None = None


def _file_uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != 'file':
        raise ValueError(f'Expected a local file artifact URI; found {uri}.')
    path = url2pathname(unquote(parsed.path))
    if parsed.netloc:
        path = f'//{parsed.netloc}{path}'
    return Path(path).resolve()


def configure_v2_mlflow(
    *,
    tracking_db_path: Path,
    artifact_root: Path,
    experiment_name: str,
) -> str:
    '''Configure a local SQLite experiment without relocating existing data.'''
    database = Path(tracking_db_path).resolve()
    artifacts = Path(artifact_root).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f'sqlite:///{database.as_posix()}')
    requested_artifact_uri = artifacts.as_uri()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=requested_artifact_uri,
        )
    else:
        existing = _file_uri_to_path(experiment.artifact_location)
        if existing != artifacts:
            raise ValueError(
                'Existing MLflow experiment artifact location differs from '
                f'the requested path: {experiment.artifact_location}.'
            )
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    global _CONFIGURED_EXPERIMENT_ID
    _CONFIGURED_EXPERIMENT_ID = str(experiment_id)
    return _CONFIGURED_EXPERIMENT_ID


@contextmanager
def tracked_pilot_parent(
    *,
    run_name: str,
    parameters: dict[str, Any],
    tags: dict[str, str],
    configuration: dict[str, Any],
) -> Iterator[str]:
    '''Open one parent run and persist immutable campaign metadata.'''
    if _CONFIGURED_EXPERIMENT_ID is None:
        raise RuntimeError('configure_v2_mlflow must be called first.')
    client = mlflow.MlflowClient()
    run = client.create_run(
        _CONFIGURED_EXPERIMENT_ID,
        tags={**tags, MLFLOW_RUN_NAME: run_name},
    )
    run_id = run.info.run_id
    for key, value in parameters.items():
        if value is not None:
            client.log_param(run_id, key, value)
    client.log_dict(run_id, configuration, 'configuration.json')
    configuration_payload = json.dumps(
        configuration, sort_keys=True, separators=(',', ':')
    ).encode('utf-8')
    client.log_param(
        run_id,
        'configuration_sha256',
        hashlib.sha256(configuration_payload).hexdigest(),
    )
    client.set_tag(run_id, 'experiment_status', 'running')
    try:
        yield run_id
    except BaseException:
        client.set_tag(run_id, 'experiment_status', 'failed')
        client.set_terminated(
            run_id, status=RunStatus.to_string(RunStatus.FAILED)
        )
        raise
    else:
        client.set_tag(run_id, 'experiment_status', 'complete')
        client.set_terminated(
            run_id, status=RunStatus.to_string(RunStatus.FINISHED)
        )


@contextmanager
def tracked_engine_child(
    *,
    parent_run_id: str,
    engine: str,
    engine_version: str,
    parameters: dict[str, Any],
    tags: dict[str, str],
) -> Iterator[str]:
    '''Open one explicit child and preserve the original engine exception.'''
    child_tags = {
        **tags,
        'model_family': engine,
        'model_version': engine_version,
        'experiment_status': 'running',
    }
    with mlflow.start_run(
        run_name=f'pilot_{engine}',
        parent_run_id=parent_run_id,
        tags=child_tags,
    ) as run:
        mlflow.log_params(parameters)
        try:
            yield run.info.run_id
        except BaseException as error:
            try:
                mlflow.set_tags(
                    {
                        'experiment_status': 'failed',
                        'error_type': type(error).__name__,
                        'error_message': str(error)[:5000],
                    }
                )
            except BaseException:
                pass
            raise


def log_engine_pilot_result(
    result: EnginePilotResult,
    *,
    data_hashes: dict[str, str],
    feature_hash: str,
) -> None:
    '''Log technical pilot diagnostics without serializing a model.'''
    probabilities = np.asarray(result.probabilities, dtype=np.float64)
    timings = {
        'early_stopping_fit_seconds': result.early_stopping_fit_seconds,
        'outer_refit_seconds': result.outer_refit_seconds,
        'prediction_seconds': result.prediction_seconds,
        'total_duration_seconds': (
            result.early_stopping_fit_seconds
            + result.outer_refit_seconds
            + result.prediction_seconds
        ),
        'peak_rss_gib': result.peak_rss_bytes / 1024**3,
    }
    metrics = {
        **{
            f'outer_{key}': value
            for key, value in result.metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
        **timings,
        'best_iteration': result.best_iteration,
        'probability_min': float(probabilities.min()),
        'probability_q05': float(np.quantile(probabilities, 0.05)),
        'probability_mean': float(probabilities.mean()),
        'probability_std': float(probabilities.std()),
        'probability_q95': float(np.quantile(probabilities, 0.95)),
        'probability_max': float(probabilities.max()),
    }
    mlflow.log_metrics(
        {key: float(value) for key, value in metrics.items()}
    )
    mlflow.log_params(
        {
            'outer_fold_id': result.outer_fold_id,
            'internal_validation_fold_id': (
                result.internal_validation_fold_id
            ),
            'classification_threshold': 0.5,
            'feature_columns_sha256': feature_hash,
            **data_hashes,
            'probabilities_sha256': hashlib.sha256(
                probabilities.tobytes()
            ).hexdigest(),
        }
    )
    counts, edges = np.histogram(
        probabilities, bins=np.linspace(0.0, 1.0, 21)
    )
    mlflow.log_dict(
        {
            'scope': 'single_outer_fold_technical_calibration_only',
            'global_oof_metric': False,
            'selection_authorized': False,
            'model_serialized': False,
            'estimator_objects_are_distinct': (
                result.estimator_objects_are_distinct
            ),
            'preprocessing_objects_are_distinct': (
                result.preprocessing_objects_are_distinct
            ),
            'probability_histogram': [
                {
                    'left': float(edges[index]),
                    'right': float(edges[index + 1]),
                    'count': int(counts[index]),
                }
                for index in range(len(counts))
            ],
        },
        'technical_diagnostics.json',
    )
    mlflow.set_tag('experiment_status', 'complete')


def tracking_summary_json(
    results: list[EnginePilotResult],
    *,
    parent_run_id: str,
) -> str:
    '''Serialize the user-facing technical summary without model objects.'''
    return json.dumps(
        {
            'parent_run_id': parent_run_id,
            'scope': 'single_outer_fold_technical_calibration_only',
            'results': [
                {
                    'engine': result.engine,
                    'outer_fold_id': result.outer_fold_id,
                    'best_iteration': result.best_iteration,
                    'early_stopping_fit_seconds': (
                        result.early_stopping_fit_seconds
                    ),
                    'outer_refit_seconds': result.outer_refit_seconds,
                    'prediction_seconds': result.prediction_seconds,
                    'peak_rss_bytes': result.peak_rss_bytes,
                    'metrics': result.metrics,
                }
                for result in results
            ],
        },
        indent=2,
        sort_keys=True,
    )
