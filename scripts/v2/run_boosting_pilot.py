'''Run the six-fit FS0 technical calibration pilot, never the full matrix.'''

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import psutil

from qrt_forecasting.common.data import load_training_target
from qrt_forecasting.common.target import create_binary_target
from qrt_forecasting.v2.boosting import (
    RAM_LIMIT_BYTES,
    LightGBMAdapter,
    boosting_adapters,
    run_engine_on_outer_fold,
)
from qrt_forecasting.v2.experiment_tracking import (
    configure_v2_mlflow,
    log_engine_pilot_result,
    tracked_engine_child,
    tracked_pilot_parent,
    tracking_summary_json,
)
from qrt_forecasting.v2.feature_cache import load_cached_feature_set
from qrt_forecasting.v2.feature_registry import (
    FEATURE_BLOCKS,
    FEATURE_SETS,
    feature_columns_sha256,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIRECTORY = (
    REPOSITORY_ROOT / 'artifacts' / 'features' / 'v2_development'
)
TRACKING_DATABASE = (
    REPOSITORY_ROOT / 'artifacts' / 'mlflow' / 'v2_foundation' / 'mlflow.db'
)
ARTIFACT_ROOT = (
    REPOSITORY_ROOT / 'artifacts' / 'mlflow' / 'v2_foundation' / 'artifacts'
)
SUMMARY_PATH = (
    REPOSITORY_ROOT / 'artifacts' / 'experiments'
    / 'BOOSTING_FOUNDATION_PILOT' / 'summary.json'
)
FEATURE_SET = 'FS0_RET_RAW'
OUTER_FOLD_ID = 0
SEED = 42
MAX_ESTIMATORS = 400
EARLY_STOPPING_PATIENCE = 20
ENGINE_TIME_LIMIT_SECONDS = 3600.0
TOTAL_TIME_LIMIT_SECONDS = 7200.0


class PeakRssMonitor:
    '''Sample process RSS and flag a sustained 12 GiB violation.'''

    def __init__(self, *, interval_seconds: float = 0.25) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = 0
        self.limit_exceeded_for_seconds = 0.0
        self._stop = threading.Event()

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(self.interval_seconds):
            rss = process.memory_info().rss
            self.peak_bytes = max(self.peak_bytes, rss)
            if rss > RAM_LIMIT_BYTES:
                self.limit_exceeded_for_seconds += self.interval_seconds
            else:
                self.limit_exceeded_for_seconds = 0.0

    def __enter__(self) -> 'PeakRssMonitor':
        self.peak_bytes = psutil.Process().memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _git_metadata() -> dict[str, str]:
    def command(*arguments: str) -> str:
        return subprocess.run(
            ['git', *arguments],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        'git_commit': command('rev-parse', 'HEAD'),
        'git_branch': command('branch', '--show-current'),
        'git_worktree_status': (
            'clean' if not command('status', '--porcelain') else 'dirty'
        ),
    }


def _measure_lightgbm_layout(matrix: pd.DataFrame) -> dict[str, object]:
    '''Choose one exclusive layout from a short, model-free memory-copy timing.'''
    sample = matrix.iloc[: min(50_000, len(matrix))].to_numpy(
        dtype=np.float32
    )
    started = time.perf_counter()
    np.asfortranarray(sample)
    column_seconds = time.perf_counter() - started
    started = time.perf_counter()
    np.ascontiguousarray(sample)
    row_seconds = time.perf_counter() - started
    return {
        'force_col_wise': column_seconds <= row_seconds,
        'force_row_wise': row_seconds < column_seconds,
        'column_copy_seconds': column_seconds,
        'row_copy_seconds': row_seconds,
    }


def _load_target(identity: pd.DataFrame) -> np.ndarray:
    target = load_training_target(usecols=['ROW_ID', 'target'])
    if target['ROW_ID'].isna().any() or target['ROW_ID'].duplicated().any():
        raise ValueError('Training target ROW_ID values must be present and unique.')
    selected = identity[['ROW_ID']].merge(
        target, on='ROW_ID', how='left', validate='one_to_one'
    )
    if selected['target'].isna().any():
        raise ValueError('A development target is missing.')
    return create_binary_target(selected['target']).to_numpy(dtype=np.int8)


def main() -> None:
    identity, matrix, cache_manifest = load_cached_feature_set(
        CACHE_DIRECTORY, FEATURE_SET
    )
    if len(identity) != 421_654:
        raise ValueError('The pilot requires exactly 421,654 development rows.')
    target = _load_target(identity)
    numeric_columns = list(matrix.columns)
    categorical_columns: list[str] = []
    layout_measurement = _measure_lightgbm_layout(matrix)
    metadata = _git_metadata()
    configure_v2_mlflow(
        tracking_db_path=TRACKING_DATABASE,
        artifact_root=ARTIFACT_ROOT,
        experiment_name='qrt_v2_boosting_foundation',
    )
    configuration = {
        'scope': 'single_outer_fold_technical_calibration_only',
        'feature_set': FEATURE_SET,
        'outer_fold_id': OUTER_FOLD_ID,
        'seed': SEED,
        'max_estimators': MAX_ESTIMATORS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'threshold': 0.5,
        'lightgbm_layout_measurement': layout_measurement,
        'ram_limit_bytes': RAM_LIMIT_BYTES,
        'engine_time_limit_seconds': ENGINE_TIME_LIMIT_SECONDS,
        'total_time_limit_seconds': TOTAL_TIME_LIMIT_SECONDS,
        'full_matrix_started': False,
        'optuna_started': False,
        'lockbox_metrics_computed': False,
    }
    results = []
    campaign_started = time.perf_counter()
    with tracked_pilot_parent(
        run_name='BOOSTING_FOUNDATION_PILOT',
        parameters={
            'feature_set': FEATURE_SET,
            'outer_fold_id': OUTER_FOLD_ID,
            'seed': SEED,
            'n_engines': 3,
            'maximum_real_fits': 6,
        },
        tags={**metadata, 'validation_scope': 'development_single_fold'},
        configuration=configuration,
    ) as parent_run_id:
        for adapter in boosting_adapters(
            max_estimators=MAX_ESTIMATORS,
            patience=EARLY_STOPPING_PATIENCE,
        ):
            if isinstance(adapter, LightGBMAdapter):
                adapter.force_col_wise = bool(
                    layout_measurement['force_col_wise']
                )
            engine_started = time.perf_counter()
            with tracked_engine_child(
                parent_run_id=parent_run_id,
                engine=adapter.name,
                engine_version=adapter.version,
                parameters={
                    'seed': SEED,
                    'max_estimators': MAX_ESTIMATORS,
                    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
                    'cpu_only': True,
                },
                tags={'feature_set': FEATURE_SET},
            ):
                with PeakRssMonitor() as memory:
                    result = run_engine_on_outer_fold(
                        adapter,
                        matrix,
                        target,
                        identity,
                        outer_fold_id=OUTER_FOLD_ID,
                        numeric_columns=numeric_columns,
                        categorical_columns=categorical_columns,
                        peak_rss_bytes=0,
                    )
                result.peak_rss_bytes = memory.peak_bytes
                if memory.limit_exceeded_for_seconds >= 5.0:
                    raise MemoryError('Sustained 12 GiB RAM limit exceeded.')
                engine_seconds = time.perf_counter() - engine_started
                if engine_seconds > ENGINE_TIME_LIMIT_SECONDS:
                    raise TimeoutError(
                        f'{adapter.name} exceeded its 60-minute pilot budget.'
                    )
                log_engine_pilot_result(
                    result,
                    data_hashes={
                        'source_sha256': cache_manifest['source_sha256'],
                        'row_ids_sha256': cache_manifest['row_ids_sha256'],
                        'folds_sha256': cache_manifest['folds_sha256'],
                    },
                    feature_hash=feature_columns_sha256(tuple(matrix.columns)),
                )
                results.append(result)
        if time.perf_counter() - campaign_started > TOTAL_TIME_LIMIT_SECONDS:
            raise TimeoutError('The pilot exceeded its two-hour total budget.')
        mlflow.MlflowClient().log_metric(
            parent_run_id, 'completed_engine_count', len(results)
        )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = tracking_summary_json(results, parent_run_id=parent_run_id)
    SUMMARY_PATH.write_text(summary + '\n', encoding='utf-8', newline='\n')
    print(summary)
    print(f'tracking_database={TRACKING_DATABASE.resolve()}')
    print(f'artifact_root={ARTIFACT_ROOT.resolve()}')


if __name__ == '__main__':
    main()
