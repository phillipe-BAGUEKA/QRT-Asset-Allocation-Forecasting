'''Strict construction and persistence of a Challenge Data submission.'''

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def infer_prediction_column(sample_submission: pd.DataFrame) -> str:
    '''Infer the unique non-ROW_ID prediction column from the official sample.'''
    if 'ROW_ID' not in sample_submission.columns:
        raise ValueError('sample_submission is missing required column ROW_ID.')
    prediction_columns = [
        column for column in sample_submission.columns if column != 'ROW_ID'
    ]
    if len(prediction_columns) != 1:
        raise ValueError(
            'sample_submission must contain exactly one prediction column.'
        )
    return prediction_columns[0]


def validate_reference_gate(
    artifact_directory: Path,
    *,
    experiment_id: str,
    configuration_sha256: str,
    assignment_sha256: str,
) -> dict[str, Any]:
    '''Reject submission work unless the persisted local OOF gate is valid.'''
    artifact_directory = Path(artifact_directory).resolve()
    gate_path = artifact_directory / 'local_gate.json'
    summary_path = artifact_directory / 'summary.json'
    if not gate_path.is_file() or not summary_path.is_file():
        raise ValueError('Required OOF gate artifacts are missing.')
    with gate_path.open(encoding='utf-8') as gate_file:
        gate = json.load(gate_file)
    with summary_path.open(encoding='utf-8') as summary_file:
        summary = json.load(summary_file)
    if gate.get('technical_status') != 'PASS' or gate.get('reproducible') is not True:
        raise ValueError('The local OOF technical gate did not pass.')
    if summary.get('gate') != gate:
        raise ValueError('Gate and OOF summary disagree.')
    expected = {
        'experiment_id': experiment_id,
        'configuration_sha256': configuration_sha256,
        'assignment_sha256': assignment_sha256,
        'evaluation_scope': 'development_oof_only',
        'lockbox_metrics_computed': False,
    }
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise ValueError(f'OOF summary field {key} is invalid.')
    if summary.get('n_development_rows') != 421654:
        raise ValueError('OOF summary development row count is invalid.')
    return summary


def fit_full_training_pipeline(
    training_data: pd.DataFrame,
    *,
    feature_columns: list[str],
    pipeline_factory: Any,
    expected_n_rows: int,
) -> tuple[Pipeline, float]:
    '''Fit one fresh pipeline on all labelled rows after strict validation.'''
    required = {'ROW_ID', 'class', *feature_columns}
    missing = sorted(required.difference(training_data.columns))
    if missing:
        raise ValueError(f'Full training data is missing columns: {missing}.')
    if len(training_data) != expected_n_rows:
        raise ValueError(
            f'Expected {expected_n_rows} full training rows; found '
            f'{len(training_data)}.'
        )
    if training_data['ROW_ID'].isna().any() or training_data['ROW_ID'].duplicated().any():
        raise ValueError('Full training ROW_ID values must be present and unique.')
    if set(training_data['class'].unique().tolist()) != {0, 1}:
        raise ValueError('Full training class must contain exactly 0 and 1.')
    pipeline = pipeline_factory()
    if not isinstance(pipeline, Pipeline):
        raise TypeError('pipeline_factory must return an sklearn Pipeline.')
    started = time.perf_counter()
    pipeline.fit(training_data[feature_columns], training_data['class'])
    return pipeline, time.perf_counter() - started


def positive_class_probabilities(
    fitted_pipeline: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    '''Extract finite probabilities for class 1 from a fitted pipeline.'''
    probabilities = np.asarray(fitted_pipeline.predict_proba(features), dtype=float)
    classes = np.asarray(fitted_pipeline.classes_)
    positions = np.flatnonzero(classes == 1)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
        raise ValueError('The fitted pipeline returned an invalid probability shape.')
    if len(positions) != 1:
        raise ValueError('The fitted pipeline must expose class 1 exactly once.')
    positive = probabilities[:, int(positions[0])]
    if not np.isfinite(positive).all():
        raise ValueError('Test probabilities must be finite.')
    if ((positive < 0.0) | (positive > 1.0)).any():
        raise ValueError('Test probabilities must lie in [0, 1].')
    return positive


def build_submission_frame(
    test_features: pd.DataFrame,
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
) -> tuple[pd.DataFrame, str]:
    '''Build a schema-exact binary submission in official ROW_ID order.'''
    if 'ROW_ID' not in test_features.columns:
        raise ValueError('X_test is missing required column ROW_ID.')
    if 'ROW_ID' not in sample_submission.columns:
        raise ValueError('sample_submission is missing required column ROW_ID.')
    for frame, name in (
        (test_features, 'X_test'),
        (sample_submission, 'sample_submission'),
    ):
        if frame['ROW_ID'].isna().any():
            raise ValueError(f'{name} contains missing ROW_ID values.')
        if frame['ROW_ID'].duplicated().any():
            raise ValueError(f'{name} contains duplicate ROW_ID values.')
    if len(test_features) != len(sample_submission):
        raise ValueError('X_test and sample_submission row counts differ.')
    if not test_features['ROW_ID'].reset_index(drop=True).equals(
        sample_submission['ROW_ID'].reset_index(drop=True)
    ):
        raise ValueError('Official ROW_ID values or order differ between inputs.')
    values = np.asarray(predictions)
    if values.ndim != 1 or len(values) != len(test_features):
        raise ValueError('Predictions must match the X_test row count.')
    if pd.isna(values).any() or set(np.unique(values).tolist()).difference({0, 1}):
        raise ValueError('Submission predictions must contain only 0 and 1.')

    prediction_column = infer_prediction_column(sample_submission)
    submission = sample_submission.copy()
    submission[prediction_column] = values.astype('int8')
    if submission.columns.tolist() != sample_submission.columns.tolist():
        raise ValueError('Submission columns or order changed.')
    if submission.isna().any().any():
        raise ValueError('Submission contains missing values.')
    return submission, prediction_column


def dataframe_row_id_sha256(frame: pd.DataFrame) -> str:
    '''Hash ROW_ID values in their current order.'''
    if 'ROW_ID' not in frame.columns:
        raise ValueError('Cannot hash missing ROW_ID column.')
    payload = frame[['ROW_ID']].to_csv(index=False, lineterminator='\n').encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def persist_validated_submission(
    submission: pd.DataFrame,
    path: Path,
) -> str:
    '''Atomically write, re-read and hash a validated submission CSV.'''
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', newline='', suffix='.csv',
        dir=path.parent, delete=False
    ) as temporary:
        submission.to_csv(temporary, index=False, lineterminator='\n')
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
    reloaded = pd.read_csv(path)
    pd.testing.assert_frame_equal(
        submission.reset_index(drop=True),
        reloaded.astype(submission.dtypes.to_dict()).reset_index(drop=True),
        check_dtype=True,
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest:
        raise ValueError('Submission SHA-256 could not be computed.')
    return digest
