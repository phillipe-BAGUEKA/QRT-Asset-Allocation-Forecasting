'''Evaluate the frozen lockbox once, refit, and build final local artifacts.'''

from __future__ import annotations

import gc
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from qrt_forecasting.common.data import load_training_features, load_training_target
from qrt_forecasting.common.metrics import binary_classification_metrics
from qrt_forecasting.common.target import build_binary_training_frame
from qrt_forecasting.v2.final_model import (
    FINAL_FEATURE_COLUMNS,
    FINAL_MODEL_NAME,
    FINAL_MODEL_VERSION,
    FINAL_RANDOM_STATE,
    FINAL_THRESHOLD,
    build_final_gradient_boosting_pipeline,
    partition_final_training_data,
    persist_joblib_atomic,
    persist_json_atomic,
    validate_frozen_final_configuration,
)
from qrt_forecasting.v2.folds import load_assignment_artifacts
from qrt_forecasting.v2.gradient_boosting import file_sha256
from qrt_forecasting.v2.submission import (
    build_submission_frame,
    dataframe_row_id_sha256,
    persist_validated_submission,
    positive_class_probabilities,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIRECTORY = REPOSITORY_ROOT / 'data' / 'raw'
FINAL_CONFIG_PATH = REPOSITORY_ROOT / 'configs' / 'final' / 'GRADIENT_BOOSTING_RET20_FINAL.toml'
REFERENCE_CONFIG_PATH = REPOSITORY_ROOT / 'configs' / 'experiments' / 'GB_RET20_REFERENCE.toml'
REFERENCE_REPORT_PATH = REPOSITORY_ROOT / 'reports' / 'experiments' / 'GB_RET20_REFERENCE.json'
ASSIGNMENT_PATH = REPOSITORY_ROOT / 'artifacts' / 'folds' / 'v2_grouped_assignment.csv'
ASSIGNMENT_MANIFEST_PATH = REPOSITORY_ROOT / 'reports' / 'validation' / 'v2_grouped_folds_manifest.json'
MODEL_PATH = REPOSITORY_ROOT / 'models' / f'{FINAL_MODEL_NAME}.joblib'
MODEL_MANIFEST_PATH = REPOSITORY_ROOT / 'models' / f'{FINAL_MODEL_NAME}.manifest.json'
FINAL_REPORT_PATH = REPOSITORY_ROOT / 'reports' / 'final' / f'{FINAL_MODEL_NAME}.json'
SUBMISSION_PATH = REPOSITORY_ROOT / 'data' / 'submissions' / f'{FINAL_MODEL_NAME}.csv'
SUBMISSION_MANIFEST_PATH = REPOSITORY_ROOT / 'data' / 'submissions' / f'{FINAL_MODEL_NAME}.manifest.json'
EXPECTED_TRAINING_ROWS = 527073
EXPECTED_DEVELOPMENT_ROWS = 421654
EXPECTED_LOCKBOX_ROWS = 105419


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ['git', *arguments], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f'Git returned no value for {arguments!r}.')
    return value


def _validate_clean_main() -> tuple[str, str]:
    branch = _git_value('branch', '--show-current')
    commit = _git_value('rev-parse', 'HEAD')
    if branch != 'main':
        raise ValueError(f'Finalization must run on main, not {branch!r}.')
    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout
    if status:
        raise ValueError('Finalization requires a clean Git worktree and index.')
    return branch, commit


def _refuse_overwrite() -> None:
    outputs = [MODEL_PATH, MODEL_MANIFEST_PATH, FINAL_REPORT_PATH, SUBMISSION_PATH, SUBMISSION_MANIFEST_PATH]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            'Finalization is intentionally single-use; refusing to overwrite: '
            + ', '.join(existing)
        )


def _versions() -> dict[str, str]:
    return {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit_learn': sklearn.__version__,
        'joblib': joblib.__version__,
    }


def _load_labelled_training_data() -> pd.DataFrame:
    features = load_training_features(
        RAW_DATA_DIRECTORY,
        usecols=['ROW_ID', 'TS', *FINAL_FEATURE_COLUMNS],
    )
    target = load_training_target(RAW_DATA_DIRECTORY, usecols=['ROW_ID', 'target'])
    training_data = build_binary_training_frame(features, target)
    if len(training_data) != EXPECTED_TRAINING_ROWS:
        raise ValueError(f'Expected {EXPECTED_TRAINING_ROWS} labelled rows; found {len(training_data)}.')
    return training_data


def _validate_partition_counts(development: pd.DataFrame, lockbox: pd.DataFrame) -> None:
    if len(development) != EXPECTED_DEVELOPMENT_ROWS:
        raise ValueError('Frozen development row count changed.')
    if len(lockbox) != EXPECTED_LOCKBOX_ROWS:
        raise ValueError('Frozen lockbox row count changed.')
    if development['TS'].nunique() != 2028 or lockbox['TS'].nunique() != 494:
        raise ValueError('Frozen development or lockbox group count changed.')


def _fit_and_score_lockbox(
    development: pd.DataFrame,
    lockbox: pd.DataFrame,
) -> tuple[dict[str, Any], float, float]:
    pipeline = build_final_gradient_boosting_pipeline()
    fit_started = time.perf_counter()
    pipeline.fit(development[list(FINAL_FEATURE_COLUMNS)], development['class'])
    fit_seconds = time.perf_counter() - fit_started
    prediction_started = time.perf_counter()
    probabilities = positive_class_probabilities(
        pipeline, lockbox[list(FINAL_FEATURE_COLUMNS)]
    )
    prediction_seconds = time.perf_counter() - prediction_started
    metrics = binary_classification_metrics(
        lockbox['class'].to_numpy(), probabilities, threshold=FINAL_THRESHOLD
    )
    metrics['n_groups'] = int(lockbox['TS'].nunique())
    metrics['development_n_rows'] = int(len(development))
    metrics['development_n_groups'] = int(development['TS'].nunique())
    metrics['group_overlap_count'] = 0
    return metrics, fit_seconds, prediction_seconds


def _fit_full_model(training_data: pd.DataFrame) -> tuple[Any, float]:
    pipeline = build_final_gradient_boosting_pipeline()
    started = time.perf_counter()
    pipeline.fit(training_data[list(FINAL_FEATURE_COLUMNS)], training_data['class'])
    return pipeline, time.perf_counter() - started


def _build_submission(pipeline: Any) -> tuple[str, dict[str, Any]]:
    test_features = pd.read_csv(
        RAW_DATA_DIRECTORY / 'X_test.csv',
        usecols=['ROW_ID', *FINAL_FEATURE_COLUMNS],
    )
    sample = pd.read_csv(RAW_DATA_DIRECTORY / 'sample_submission.csv')
    probabilities = positive_class_probabilities(
        pipeline, test_features[list(FINAL_FEATURE_COLUMNS)]
    )
    predictions = (probabilities >= FINAL_THRESHOLD).astype('int8')
    submission, prediction_column = build_submission_frame(test_features, sample, predictions)
    digest = persist_validated_submission(submission, SUBMISSION_PATH)
    reloaded = pd.read_csv(SUBMISSION_PATH)
    if reloaded.columns.tolist() != sample.columns.tolist() or len(reloaded) != len(sample):
        raise ValueError('Persisted submission no longer matches the official schema.')
    manifest = {
        'schema_version': 1,
        'submission_file': str(SUBMISSION_PATH.relative_to(REPOSITORY_ROOT)),
        'submission_sha256': digest,
        'n_rows': int(len(submission)),
        'columns': submission.columns.tolist(),
        'prediction_column': prediction_column,
        'prediction_values': sorted(submission[prediction_column].unique().astype(int).tolist()),
        'positive_prediction_rate': float(submission[prediction_column].mean()),
        'row_id_sha256': dataframe_row_id_sha256(submission),
    }
    persist_json_atomic(manifest, SUBMISSION_MANIFEST_PATH)
    return digest, manifest


def main() -> None:
    '''Execute the authorized one-time finalization sequence.'''
    _refuse_overwrite()
    branch, commit = _validate_clean_main()
    configuration, reference_report, configuration_hash = validate_frozen_final_configuration(
        FINAL_CONFIG_PATH, REFERENCE_CONFIG_PATH, REFERENCE_REPORT_PATH
    )
    assignment, assignment_manifest = load_assignment_artifacts(
        ASSIGNMENT_PATH, ASSIGNMENT_MANIFEST_PATH
    )
    if assignment_manifest['assignment_sha256'] != configuration['validation']['assignment_sha256']:
        raise ValueError('Final configuration and assignment manifest disagree.')

    training_data = _load_labelled_training_data()
    development, lockbox = partition_final_training_data(training_data, assignment)
    _validate_partition_counts(development, lockbox)
    lockbox_metrics, lockbox_fit_seconds, lockbox_prediction_seconds = _fit_and_score_lockbox(
        development, lockbox
    )
    del development, lockbox, assignment
    gc.collect()

    pipeline, full_fit_seconds = _fit_full_model(training_data)
    training_rows = int(len(training_data))
    training_groups = int(training_data['TS'].nunique())
    del training_data
    gc.collect()
    model_sha256 = persist_joblib_atomic(pipeline, MODEL_PATH)
    reloaded_pipeline = joblib.load(MODEL_PATH)
    if list(reloaded_pipeline.feature_names_in_) != list(FINAL_FEATURE_COLUMNS):
        raise ValueError('Reloaded model input schema changed.')
    submission_sha256, submission_manifest = _build_submission(reloaded_pipeline)

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        'schema_version': 1,
        'model_name': FINAL_MODEL_NAME,
        'model_version': FINAL_MODEL_VERSION,
        'generated_at_utc': generated_at,
        'git_commit': commit,
        'git_branch': branch,
        'model_family': 'GradientBoostingClassifier',
        'feature_columns': list(FINAL_FEATURE_COLUMNS),
        'feature_count': len(FINAL_FEATURE_COLUMNS),
        'pipeline': ['OrderedColumnSelector', 'SimpleImputer', 'GradientBoostingClassifier'],
        'preprocessing': configuration['preprocessing'],
        'model_parameters': {**configuration['model'], 'random_state': FINAL_RANDOM_STATE},
        'prediction_threshold': FINAL_THRESHOLD,
        'random_state': FINAL_RANDOM_STATE,
        'training_scope': 'full_train',
        'training_row_count': training_rows,
        'training_group_count': training_groups,
        'training_date_utc': generated_at,
        'configuration_sha256': configuration_hash,
        'artifact_sha256': model_sha256,
        'development_oof_metrics': reference_report['global_metrics'],
        'lockbox_metrics': lockbox_metrics,
        'versions': _versions(),
        'python_version': platform.python_version(),
        'scikit_learn_version': sklearn.__version__,
        'joblib_version': joblib.__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
    }
    persist_json_atomic(manifest, MODEL_MANIFEST_PATH)
    report = {
        'schema_version': 1,
        'status': 'portfolio_model_finalized',
        'lockbox_used_once': True,
        'selection_scope_before_lockbox': 'development_oof_only',
        'configuration_sha256': configuration_hash,
        'assignment_sha256': assignment_manifest['assignment_sha256'],
        'development_oof_metrics': reference_report['global_metrics'],
        'lockbox_metrics': lockbox_metrics,
        'timing': {
            'lockbox_fit_seconds': lockbox_fit_seconds,
            'lockbox_prediction_seconds': lockbox_prediction_seconds,
            'full_train_fit_seconds': full_fit_seconds,
        },
        'model_path': str(MODEL_PATH.relative_to(REPOSITORY_ROOT)),
        'model_sha256': model_sha256,
        'submission_path': str(SUBMISSION_PATH.relative_to(REPOSITORY_ROOT)),
        'submission_sha256': submission_sha256,
        'submission': submission_manifest,
    }
    persist_json_atomic(report, FINAL_REPORT_PATH)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
