'''Evaluate the frozen lockbox once, refit, and build final local artifacts.'''

from __future__ import annotations

import gc
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

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
LOCKBOX_REPORT_PATH = REPOSITORY_ROOT / 'reports' / 'final' / f'{FINAL_MODEL_NAME}.lockbox.json'
SUBMISSION_PATH = REPOSITORY_ROOT / 'data' / 'submissions' / f'{FINAL_MODEL_NAME}.csv'
SUBMISSION_MANIFEST_PATH = REPOSITORY_ROOT / 'data' / 'submissions' / f'{FINAL_MODEL_NAME}.manifest.json'
EXPECTED_TRAINING_ROWS = 527073
EXPECTED_DEVELOPMENT_ROWS = 421654
EXPECTED_LOCKBOX_ROWS = 105419
TRAINING_CSV_SEPARATOR = ','
TEST_CSV_SEPARATOR = ';'
SAMPLE_SUBMISSION_CSV_SEPARATOR = ','


def _git_value(*arguments: str) -> str:
    completed = subprocess.run(
        ['git', *arguments], cwd=REPOSITORY_ROOT, check=True,
        capture_output=True, text=True,
    )
    value = completed.stdout.strip()
    if not value:
        raise ValueError(f'Git returned no value for {arguments!r}.')
    return value


def _validate_main() -> tuple[str, str]:
    branch = _git_value('branch', '--show-current')
    commit = _git_value('rev-parse', 'HEAD')
    if branch != 'main':
        raise ValueError(f'Finalization must run on main, not {branch!r}.')
    return branch, commit


def _refuse_overwrite() -> None:
    outputs = [MODEL_MANIFEST_PATH, FINAL_REPORT_PATH, LOCKBOX_REPORT_PATH, SUBMISSION_PATH, SUBMISSION_MANIFEST_PATH]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            'Finalization is intentionally single-use; refusing to overwrite: '
            + ', '.join(existing)
        )


def _read_csv(
    path: Path,
    *,
    separator: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    '''Read one challenge CSV with its explicit file contract.'''
    return pd.read_csv(path, sep=separator, usecols=usecols)


def _load_test_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    test_features = _read_csv(
        RAW_DATA_DIRECTORY / 'X_test.csv', separator=TEST_CSV_SEPARATOR,
        usecols=['ROW_ID', *FINAL_FEATURE_COLUMNS],
    )[['ROW_ID', *FINAL_FEATURE_COLUMNS]]
    sample = _read_csv(
        RAW_DATA_DIRECTORY / 'sample_submission.csv',
        separator=SAMPLE_SUBMISSION_CSV_SEPARATOR,
    )
    if test_features.columns.tolist() != ['ROW_ID', *FINAL_FEATURE_COLUMNS]:
        raise ValueError('X_test feature order could not be canonicalized.')
    for frame, name in ((test_features, 'X_test'), (sample, 'sample_submission')):
        if 'ROW_ID' not in frame.columns:
            raise ValueError(f'{name} is missing ROW_ID.')
        if frame['ROW_ID'].isna().any() or frame['ROW_ID'].duplicated().any():
            raise ValueError(f'{name} ROW_ID values must be present and unique.')
    if len(test_features) != len(sample) or not test_features['ROW_ID'].equals(sample['ROW_ID']):
        raise ValueError('X_test and sample_submission ROW_ID order differs.')
    return test_features, sample


def _validate_output_directories() -> None:
    '''Prove every output directory supports atomic write and re-open.'''
    for directory in {MODEL_PATH.parent, FINAL_REPORT_PATH.parent, SUBMISSION_PATH.parent}:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', newline='\n', suffix='.tmp',
            dir=directory, delete=False,
        ) as temporary:
            temporary.write('preflight\n')
            temporary_path = Path(temporary.name)
        try:
            if temporary_path.read_text(encoding='utf-8') != 'preflight\n':
                raise ValueError(f'Output preflight failed for {directory}.')
        finally:
            temporary_path.unlink(missing_ok=True)


def _preflight_submission_contract() -> None:
    '''Validate test parsing and submission I/O before any lockbox access.'''
    test_features, sample = _load_test_inputs()
    submission, _ = build_submission_frame(
        test_features, sample, np.zeros(len(test_features), dtype='int8')
    )
    with tempfile.TemporaryDirectory(dir=SUBMISSION_PATH.parent) as directory:
        temporary_submission = Path(directory) / 'submission.csv'
        persist_validated_submission(submission, temporary_submission)
        reopened = _read_csv(
            temporary_submission, separator=SAMPLE_SUBMISSION_CSV_SEPARATOR
        )
        if reopened.columns.tolist() != sample.columns.tolist() or len(reopened) != len(sample):
            raise ValueError('Temporary submission round-trip failed.')
    _validate_output_directories()


def _frozen_specification_hash(
    configuration_hash: str,
    assignment_sha256: str,
) -> tuple[dict[str, Any], str]:
    classifier = build_final_gradient_boosting_pipeline().named_steps['classifier']
    parameter_names = [
        'learning_rate', 'n_estimators', 'max_depth', 'min_samples_leaf',
        'subsample', 'max_features', 'random_state',
    ]
    specification = {
        'configuration_sha256': configuration_hash,
        'assignment_sha256': assignment_sha256,
        'model_class': type(classifier).__name__,
        'features': list(FINAL_FEATURE_COLUMNS),
        'parameters': {name: classifier.get_params()[name] for name in parameter_names},
        'threshold': FINAL_THRESHOLD,
        'seed': FINAL_RANDOM_STATE,
    }
    payload = json.dumps(
        specification, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return specification, hashlib.sha256(payload).hexdigest()


def _versions() -> dict[str, str]:
    return {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'scikit_learn': sklearn.__version__,
        'joblib': joblib.__version__,
    }


def _load_labelled_training_data() -> pd.DataFrame:
    features = _read_csv(
        RAW_DATA_DIRECTORY / 'X_train.csv',
        separator=TRAINING_CSV_SEPARATOR,
        usecols=['ROW_ID', 'TS', *FINAL_FEATURE_COLUMNS],
    )
    target = _read_csv(
        RAW_DATA_DIRECTORY / 'y_train.csv',
        separator=TRAINING_CSV_SEPARATOR,
        usecols=['ROW_ID', 'target'],
    )
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


def _validate_existing_full_model() -> tuple[Any, str]:
    '''Validate the complete full-train model produced by the first attempt.'''
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('The first attempt full-train model is missing.')
    pipeline = joblib.load(MODEL_PATH)
    if list(getattr(pipeline, 'feature_names_in_', [])) != list(FINAL_FEATURE_COLUMNS):
        raise ValueError('Existing final model feature schema is invalid.')
    if list(getattr(pipeline, 'classes_', [])) != [0, 1]:
        raise ValueError('Existing final model fitted classes are invalid.')
    if list(pipeline.named_steps) != ['feature_selection', 'imputer', 'classifier']:
        raise ValueError('Existing final model pipeline steps are invalid.')
    classifier = pipeline.named_steps['classifier']
    expected_classifier = build_final_gradient_boosting_pipeline().named_steps['classifier']
    parameter_names = [
        'learning_rate', 'n_estimators', 'max_depth', 'min_samples_leaf',
        'subsample', 'max_features', 'random_state',
    ]
    for name in parameter_names:
        if classifier.get_params()[name] != expected_classifier.get_params()[name]:
            raise ValueError(f'Existing final model parameter {name} changed.')
    if getattr(classifier, 'n_estimators_', None) != 50:
        raise ValueError('Existing final model is not fitted with 50 estimators.')
    return pipeline, file_sha256(MODEL_PATH)


def _persist_lockbox_checkpoint(
    *,
    metrics: dict[str, Any],
    fit_seconds: float,
    prediction_seconds: float,
    specification: dict[str, Any],
    specification_sha256: str,
    model_sha256: str,
) -> tuple[dict[str, Any], str]:
    '''Atomically persist and verify the sole recorded lockbox evaluation.'''
    report = {
        'schema_version': 1,
        'status': 'lockbox_evaluation_recorded',
        'lockbox_metrics': metrics,
        'lockbox_fit_seconds': fit_seconds,
        'lockbox_prediction_seconds': prediction_seconds,
        'frozen_specification': specification,
        'frozen_specification_sha256': specification_sha256,
        'full_train_model_sha256_before_rerun': model_sha256,
        'lockbox_computation_attempts': 2,
        'lockbox_recorded_evaluations': 1,
        'first_attempt_metrics_observed': False,
        'model_changed_between_attempts': False,
        'features_changed_between_attempts': False,
        'hyperparameters_changed_between_attempts': False,
        'threshold_changed_between_attempts': False,
        'rerun_reason': 'metrics lost after downstream X_test delimiter failure',
        'lockbox_used_for_additional_model_selection': False,
    }
    persist_json_atomic(report, LOCKBOX_REPORT_PATH)
    with LOCKBOX_REPORT_PATH.open(encoding='utf-8') as source:
        reopened = json.load(source)
    if reopened != report:
        raise ValueError('Reopened lockbox report differs from recorded content.')
    if reopened['lockbox_recorded_evaluations'] != 1:
        raise ValueError('Lockbox report evaluation count is invalid.')
    return reopened, file_sha256(LOCKBOX_REPORT_PATH)


def _build_submission(pipeline: Any) -> tuple[str, dict[str, Any]]:
    test_features, sample = _load_test_inputs()
    probabilities = positive_class_probabilities(
        pipeline, test_features[list(FINAL_FEATURE_COLUMNS)]
    )
    predictions = (probabilities >= FINAL_THRESHOLD).astype('int8')
    submission, prediction_column = build_submission_frame(test_features, sample, predictions)
    digest = persist_validated_submission(submission, SUBMISSION_PATH)
    reloaded = _read_csv(
        SUBMISSION_PATH, separator=SAMPLE_SUBMISSION_CSV_SEPARATOR
    )
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
    '''Execute the authorized technical lockbox recovery and finalization.'''
    _refuse_overwrite()
    branch, commit = _validate_main()
    configuration, reference_report, configuration_hash = validate_frozen_final_configuration(
        FINAL_CONFIG_PATH, REFERENCE_CONFIG_PATH, REFERENCE_REPORT_PATH
    )
    assignment, assignment_manifest = load_assignment_artifacts(
        ASSIGNMENT_PATH, ASSIGNMENT_MANIFEST_PATH
    )
    if assignment_manifest['assignment_sha256'] != configuration['validation']['assignment_sha256']:
        raise ValueError('Final configuration and assignment manifest disagree.')
    specification, specification_sha256 = _frozen_specification_hash(
        configuration_hash, assignment_manifest['assignment_sha256']
    )

    # This preflight deliberately occurs before the only authorized rerun.
    _preflight_submission_contract()
    pipeline, model_sha256 = _validate_existing_full_model()

    training_data = _load_labelled_training_data()
    training_rows = int(len(training_data))
    training_groups = int(training_data['TS'].nunique())
    development, lockbox = partition_final_training_data(training_data, assignment)
    _validate_partition_counts(development, lockbox)
    lockbox_metrics, lockbox_fit_seconds, lockbox_prediction_seconds = _fit_and_score_lockbox(
        development, lockbox
    )
    lockbox_report, lockbox_report_sha256 = _persist_lockbox_checkpoint(
        metrics=lockbox_metrics,
        fit_seconds=lockbox_fit_seconds,
        prediction_seconds=lockbox_prediction_seconds,
        specification=specification,
        specification_sha256=specification_sha256,
        model_sha256=model_sha256,
    )
    if file_sha256(MODEL_PATH) != model_sha256:
        raise ValueError('Full-train model changed during lockbox recovery.')
    del development, lockbox, assignment, training_data
    gc.collect()
    submission_sha256, submission_manifest = _build_submission(pipeline)

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
        'full_train_fit_provenance': (
            'Completed during first attempt before the downstream X_test '
            'delimiter failure; validated and not retrained during recovery.'
        ),
        'configuration_sha256': configuration_hash,
        'frozen_specification_sha256': specification_sha256,
        'artifact_sha256': model_sha256,
        'development_oof_metrics': reference_report['global_metrics'],
        'lockbox_metrics': lockbox_metrics,
        'lockbox_report_sha256': lockbox_report_sha256,
        'lockbox_incident': {
            key: lockbox_report[key]
            for key in (
                'lockbox_computation_attempts', 'lockbox_recorded_evaluations',
                'first_attempt_metrics_observed', 'model_changed_between_attempts',
                'features_changed_between_attempts',
                'hyperparameters_changed_between_attempts',
                'threshold_changed_between_attempts', 'rerun_reason',
                'lockbox_used_for_additional_model_selection',
            )
        },
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
        'lockbox_computation_attempts': 2,
        'lockbox_recorded_evaluations': 1,
        'selection_scope_before_lockbox': 'development_oof_only',
        'configuration_sha256': configuration_hash,
        'assignment_sha256': assignment_manifest['assignment_sha256'],
        'development_oof_metrics': reference_report['global_metrics'],
        'lockbox_metrics': lockbox_metrics,
        'lockbox_report_path': str(LOCKBOX_REPORT_PATH.relative_to(REPOSITORY_ROOT)),
        'lockbox_report_sha256': lockbox_report_sha256,
        'incident': manifest['lockbox_incident'],
        'timing': {
            'lockbox_fit_seconds': lockbox_fit_seconds,
            'lockbox_prediction_seconds': lockbox_prediction_seconds,
            'full_train_fit_seconds': None,
            'full_train_fit_timing_note': 'Lost with first attempt process output.',
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
