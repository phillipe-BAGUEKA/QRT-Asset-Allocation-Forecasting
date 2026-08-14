'''Build the gated full-train submission for the best V2 feature family.'''

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
from qrt_forecasting.v2.feature_families import (
    feature_set_by_id,
    build_feature_family_pipeline,
)
from qrt_forecasting.v2.folds import load_assignment_artifacts
from qrt_forecasting.v2.gradient_boosting import file_sha256
from qrt_forecasting.v2.submission import (
    build_submission_frame,
    dataframe_row_id_sha256,
    fit_full_training_pipeline,
    persist_validated_submission,
    positive_class_probabilities,
    validate_feature_study_gate,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDY_CONFIG_PATH = (
    REPOSITORY_ROOT / 'configs' / 'experiments' / 'GB_FEATURE_FAMILY_STUDY.toml'
)
STUDY_ARTIFACT_DIRECTORY = (
    REPOSITORY_ROOT / 'artifacts' / 'experiments' / 'GB_FEATURE_FAMILY_STUDY'
)
ASSIGNMENT_PATH = REPOSITORY_ROOT / 'artifacts' / 'folds' / 'v2_grouped_assignment.csv'
FOLD_MANIFEST_PATH = (
    REPOSITORY_ROOT / 'reports' / 'validation' / 'v2_grouped_folds_manifest.json'
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
    test = pd.read_csv(
        RAW_DATA_DIRECTORY / 'X_test.csv', usecols=['ROW_ID', *feature_columns]
    )
    sample = pd.read_csv(RAW_DATA_DIRECTORY / 'sample_submission.csv')
    if set(feature_columns).difference(test.columns):
        raise ValueError('X_test is missing selected candidate features.')
    return test, sample


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
    _, fold_manifest = load_assignment_artifacts(
        ASSIGNMENT_PATH, FOLD_MANIFEST_PATH
    )
    study_summary, selected_id = validate_feature_study_gate(
        STUDY_ARTIFACT_DIRECTORY,
        assignment_sha256=fold_manifest['assignment_sha256'],
    )
    tracked_report_path = (
        REPOSITORY_ROOT
        / 'reports'
        / 'experiments'
        / 'GB_FEATURE_FAMILY_STUDY.json'
    )
    with tracked_report_path.open(encoding='utf-8') as source:
        tracked_report = json.load(source)
    if study_summary != tracked_report:
        raise ValueError('Local study summary differs from the tracked report.')
    best_numeric = feature_set_by_id(
        study_summary['best_numeric_experiment_id']
    )
    selected = feature_set_by_id(selected_id, best_numeric=best_numeric)
    commit, branch = _require_clean_git_state()

    training_features = load_training_features(
        usecols=['ROW_ID', *selected.input_features]
    )
    training_target = load_training_target(usecols=['ROW_ID', 'target'])
    training_data = build_binary_training_frame(
        training_features, training_target
    )
    del training_features, training_target
    pipeline, fit_time = fit_full_training_pipeline(
        training_data,
        feature_columns=selected.input_features,
        pipeline_factory=lambda: build_feature_family_pipeline(selected),
        expected_n_rows=EXPECTED_FULL_TRAIN_ROWS,
    )
    del training_data

    test_features, sample_submission = _load_test_inputs(selected.input_features)
    prediction_started = time.perf_counter()
    probabilities = positive_class_probabilities(
        pipeline, test_features[selected.input_features]
    )
    prediction_time = time.perf_counter() - prediction_started
    predictions = (probabilities >= 0.5).astype('int8')
    submission, prediction_column = build_submission_frame(
        test_features, sample_submission, predictions
    )

    study_hash = file_sha256(STUDY_CONFIG_PATH)
    submission_id = (
        selected.experiment_id
        + '_FULL_TRAIN_'
        + f'{commit[:12]}_{study_hash[:8]}'
    )
    output_directory = (
        REPOSITORY_ROOT / 'artifacts' / 'submissions' / submission_id
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_name = f'{submission_id}.csv'
    csv_path = output_directory / csv_name
    csv_sha256 = persist_validated_submission(submission, csv_path)
    selected_result = next(
        item for item in study_summary['experiments']
        if item['experiment_id'] == selected.experiment_id
    )
    manifest = {
        'schema_version': 1,
        'submission_id': submission_id,
        'experiment_id': selected.experiment_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit': commit,
        'git_branch': branch,
        'study_configuration_sha256': study_hash,
        'study_report_sha256': file_sha256(tracked_report_path),
        'assignment_sha256': fold_manifest['assignment_sha256'],
        'development_identifiers_sha256': fold_manifest[
            'development_identifiers_sha256'
        ],
        'oof_sha256': selected_result['oof_sha256'],
        'oof_accuracy': selected_result['global_metrics']['accuracy'],
        'oof_accuracy_gain': selected_result['gate'][
            'accuracy_gain_vs_reference_artifact'
        ],
        'oof_bootstrap_ci95': [
            selected_result['bootstrap']['ci95_lower'],
            selected_result['bootstrap']['ci95_upper'],
        ],
        'training_scope': 'full_train',
        'training_row_count': EXPECTED_FULL_TRAIN_ROWS,
        'numeric_features': list(selected.numeric_features),
        'categorical_features': list(selected.categorical_features),
        'input_features': selected.input_features,
        'preprocessing': {
            'numeric': {
                'strategy': 'constant',
                'fill_value': 0.0,
                'add_missing_indicators': selected.add_missing_indicators,
            },
            'categorical': {
                'encoder': 'OneHotEncoder',
                'handle_unknown': 'ignore',
                'fit_scope': 'full_train',
            },
        },
        'prediction_threshold': 0.5,
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
