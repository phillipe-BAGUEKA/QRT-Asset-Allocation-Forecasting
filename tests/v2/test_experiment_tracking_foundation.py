from __future__ import annotations

import mlflow
import numpy as np
import pytest

from qrt_forecasting.v2.boosting import EnginePilotResult
from qrt_forecasting.v2.experiment_tracking import (
    configure_v2_mlflow,
    log_engine_pilot_result,
    tracked_engine_child,
    tracked_pilot_parent,
)


@pytest.fixture(autouse=True)
def restore_mlflow_state():
    initial_uri = mlflow.get_tracking_uri()
    while mlflow.active_run() is not None:
        mlflow.end_run()
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(initial_uri)


def _result() -> EnginePilotResult:
    return EnginePilotResult(
        engine='xgboost',
        outer_fold_id=0,
        internal_validation_fold_id=1,
        best_iteration=7,
        early_stopping_fit_seconds=1.0,
        outer_refit_seconds=2.0,
        prediction_seconds=0.1,
        peak_rss_bytes=1024,
        metrics={'accuracy': 0.5, 'roc_auc': 0.6},
        probabilities=np.array([0.2, 0.8]),
        estimator_objects_are_distinct=True,
        preprocessing_objects_are_distinct=True,
    )


def test_mlflow_parent_child_and_required_diagnostics(tmp_path) -> None:
    database = tmp_path / 'tracking' / 'mlflow.db'
    artifacts = tmp_path / 'artifacts'
    configure_v2_mlflow(
        tracking_db_path=database,
        artifact_root=artifacts,
        experiment_name='foundation',
    )
    with tracked_pilot_parent(
        run_name='parent',
        parameters={'seed': 42},
        tags={'git_commit': 'abc'},
        configuration={'feature_set': 'FS0_RET_RAW'},
    ) as parent_id:
        with tracked_engine_child(
            parent_run_id=parent_id,
            engine='xgboost',
            engine_version='3.3.0',
            parameters={'cpu_only': True},
            tags={'feature_set': 'FS0_RET_RAW'},
        ) as child_id:
            log_engine_pilot_result(
                _result(),
                data_hashes={
                    'source_sha256': 'source',
                    'row_ids_sha256': 'rows',
                    'folds_sha256': 'folds',
                },
                feature_hash='features',
            )
    parent = mlflow.get_parent_run(child_id)
    assert parent is not None
    assert parent.info.run_id == parent_id
    child = mlflow.get_run(child_id)
    parent_run = mlflow.get_run(parent_id)
    assert len(parent_run.data.params['configuration_sha256']) == 64
    assert child.info.status == 'FINISHED'
    assert child.data.tags['experiment_status'] == 'complete'
    assert child.data.tags['model_family'] == 'xgboost'
    assert child.data.params['folds_sha256'] == 'folds'
    assert len(child.data.params['probabilities_sha256']) == 64
    assert child.data.metrics['best_iteration'] == 7.0
    assert child.data.metrics['probability_mean'] == pytest.approx(0.5)
    artifact_paths = {
        item.path for item in mlflow.MlflowClient().list_artifacts(child_id)
    }
    assert 'technical_diagnostics.json' in artifact_paths


def test_failed_child_preserves_original_exception_and_failed_status(tmp_path) -> None:
    configure_v2_mlflow(
        tracking_db_path=tmp_path / 'tracking.db',
        artifact_root=tmp_path / 'artifacts',
        experiment_name='failure',
    )
    with tracked_pilot_parent(
        run_name='parent',
        parameters={},
        tags={},
        configuration={},
    ) as parent_id:
        with pytest.raises(RuntimeError, match='original'):
            with tracked_engine_child(
                parent_run_id=parent_id,
                engine='catboost',
                engine_version='1.2.10',
                parameters={},
                tags={},
            ) as child_id:
                raise RuntimeError('original')
    assert mlflow.get_run(child_id).info.status == 'FAILED'
