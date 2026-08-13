'''Build the validated full-train Gradient Boosting reference submission.'''

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrt_forecasting.common.data import load_training_features, load_training_target
from qrt_forecasting.common.target import build_binary_training_frame
from qrt_forecasting.v2.folds import load_assignment_artifacts
from qrt_forecasting.v2.gradient_boosting import (
    build_gradient_boosting_reference_pipeline,
    file_sha256,
    load_experiment_configuration,
)
from qrt_forecasting.v2.submission import (
    build_submission_frame,
    dataframe_row_id_sha256,
    fit_full_training_pipeline,
    persist_validated_submission,
    positive_class_probabilities,
    validate_reference_gate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION_PATH = (
    REPOSITORY_ROOT / 'configs' / 'experiments' / 'GB_RET20_REFERENCE.toml'
)
EXPERIMENT_ARTIFACT_DIRECTORY = (
    REPOSITORY_ROOT / 'artifacts' / 'experiments' / 'GB_RET20_REFERENCE'
)
RAW_DATA_DIRECTORY = REPOSITORY_ROOT / 'data' / 'raw'
EXPECTED_FULL_TRAIN_ROWS = 527073


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ['git', *arguments], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True, encoding='utf-8'
    )
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f'Git returned no value for {arguments}.')
    return value


def _require_clean_git_state() -> tuple[str, str]:
    commit = _git_value('rev-parse', 'HEAD')
    branch = _git_value('branch', '--show-current')
    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True, encoding='utf-8'
    ).stdout.strip()
    if status:
        raise ValueError('Submission generation requires a clean Git worktree.')
    return commit, branch


def _load_test_inputs(feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_path = RAW_DATA_DIRECTORY / 'X_test.csv'
    sample_path = RAW_DATA_DIRECTORY / 'sample_submission.csv'
    if not test_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError('Official X_test or sample_submission file is missing.')
    test_features = pd.read_csv(test_path, usecols=['ROW_ID', *feature_columns])
    sample_submission = pd.read_csv(sample_path)
    missing_features = sorted(set(feature_columns).difference(test_features.columns))
    if missing_features:
        raise ValueError(f'X_test is missing reference features: {missing_features}.')
    return test_features, sample_submission


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', newline='\n', suffix='.json',
        dir=path.parent, delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write('\n')
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def main() -> None:
    configuration = load_experiment_configuration(CONFIGURATION_PATH)
    configuration_hash = file_sha256(CONFIGURATION_PATH)
    validation = configuration['validation']
    _, frozen_manifest = load_assignment_artifacts(
        REPOSITORY_ROOT / validation['assignment_path'],
        REPOSITORY_ROOT / validation['manifest_path'],
    )
    if frozen_manifest.get('assignment_sha256') != validation['assignment_sha256']:
        raise ValueError('Current frozen assignment hash is invalid.')
    summary = validate_reference_gate(
        EXPERIMENT_ARTIFACT_DIRECTORY,
        experiment_id=configuration['experiment_id'],
        configuration_sha256=configuration_hash,
        assignment_sha256=configuration['validation']['assignment_sha256'],
    )
    commit, branch = _require_clean_git_state()

    feature_columns = configuration['feature_columns']
    training_features = load_training_features(
        usecols=['ROW_ID', *feature_columns]
    )
    training_target = load_training_target(usecols=['ROW_ID', 'target'])
    training_data = build_binary_training_frame(training_features, training_target)
    del training_features, training_target
    pipeline, fit_time = fit_full_training_pipeline(
        training_data,
        feature_columns=feature_columns,
        pipeline_factory=build_gradient_boosting_reference_pipeline,
        expected_n_rows=EXPECTED_FULL_TRAIN_ROWS,
    )
    del training_data

    test_features, sample_submission = _load_test_inputs(feature_columns)
    prediction_started = time.perf_counter()
    probabilities = positive_class_probabilities(
        pipeline, test_features[feature_columns]
    )
    prediction_time = time.perf_counter() - prediction_started
    predictions = (probabilities >= configuration['threshold']).astype('int8')
    submission, prediction_column = build_submission_frame(
        test_features, sample_submission, predictions
    )

    submission_id = (
        configuration['experiment_id']
        + '_FULL_TRAIN_'
        f'{commit[:12]}_{configuration_hash[:8]}'
    )
    output_directory = REPOSITORY_ROOT / 'artifacts' / 'submissions' / submission_id
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_name = f'{submission_id}.csv'
    csv_path = output_directory / csv_name
    csv_sha256 = persist_validated_submission(submission, csv_path)
    manifest = {
        'schema_version': 1,
        'submission_id': submission_id,
        'experiment_id': configuration['experiment_id'],
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': commit,
        'git_branch': branch,
        'configuration_sha256': configuration_hash,
        'assignment_sha256': configuration['validation']['assignment_sha256'],
        'oof_sha256': summary['oof_sha256'],
        'oof_gate_status': summary['gate']['technical_status'],
        'training_scope': 'full_train',
        'training_row_count': EXPECTED_FULL_TRAIN_ROWS,
        'features': feature_columns,
        'preprocessing': configuration['preprocessing'],
        'model_parameters': {
            **configuration['model'],
            'random_state': configuration['random_state'],
        },
        'prediction_threshold': configuration['threshold'],
        'test_row_count': int(len(test_features)),
        'test_row_ids_sha256': dataframe_row_id_sha256(test_features),
        'prediction_column': prediction_column,
        'predicted_positive_count': int(predictions.sum()),
        'predicted_positive_rate': float(predictions.mean()),
        'probability_min': float(probabilities.min()),
        'probability_max': float(probabilities.max()),
        'probability_mean': float(probabilities.mean()),
        'probability_std': float(probabilities.std()),
        'fit_time_seconds': fit_time,
        'prediction_time_seconds': prediction_time,
        'csv_name': csv_name,
        'csv_path': str(csv_path.resolve()),
        'csv_sha256': csv_sha256,
        'model_persisted': False,
        'lockbox_metrics_computed': False,
        'uploaded': False,
    }
    manifest_path = output_directory / f'{submission_id}.manifest.json'
    _atomic_json_write(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
