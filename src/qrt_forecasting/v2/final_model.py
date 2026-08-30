'''Frozen final Gradient Boosting pipeline and validation helpers.'''

from __future__ import annotations

import json
import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from qrt_forecasting.v2.folds import DEVELOPMENT_ROLE, FOLD_COLUMN, LOCKBOX_ROLE, ROLE_COLUMN
from qrt_forecasting.v2.gradient_boosting import (
    REFERENCE_FEATURE_COLUMNS,
    REFERENCE_MODEL_PARAMETERS,
    REFERENCE_RANDOM_STATE,
    REFERENCE_THRESHOLD,
    file_sha256,
    load_experiment_configuration,
    sha256_canonical_text,
)


FINAL_MODEL_NAME = 'gradient_boosting_ret20_final'
FINAL_MODEL_VERSION = '2.0.0'
FINAL_FEATURE_COLUMNS = tuple(REFERENCE_FEATURE_COLUMNS)
FINAL_THRESHOLD = REFERENCE_THRESHOLD
FINAL_RANDOM_STATE = REFERENCE_RANDOM_STATE


class OrderedColumnSelector(TransformerMixin, BaseEstimator):
    '''Select the frozen input schema in deterministic order.'''

    def __init__(self, columns: tuple[str, ...]) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> 'OrderedColumnSelector':
        del y
        self._validate(X)
        self.n_features_in_ = len(X.columns)
        self.feature_names_in_ = X.columns.to_numpy(dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        return X.loc[:, list(self.columns)].copy()

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        del input_features
        return pd.Index(self.columns).to_numpy(dtype=object)

    def _validate(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Final model input must be a pandas DataFrame.')
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise ValueError(f'Final model input is missing columns: {missing}.')


def build_final_gradient_boosting_pipeline() -> Pipeline:
    '''Return a fresh pipeline containing selection, imputation and classifier.'''
    return Pipeline(
        steps=[
            ('feature_selection', OrderedColumnSelector(FINAL_FEATURE_COLUMNS)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
            ('classifier', GradientBoostingClassifier(**REFERENCE_MODEL_PARAMETERS)),
        ]
    )


def validate_frozen_final_configuration(
    final_configuration_path: Path,
    reference_configuration_path: Path,
    reference_report_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    '''Validate the frozen decision and prove lockbox metrics were unused.'''
    reference = load_experiment_configuration(reference_configuration_path)
    with Path(final_configuration_path).open('rb') as source:
        final_configuration = tomllib.load(source)
    with Path(reference_report_path).open(encoding='utf-8') as source:
        reference_report = json.load(source)

    expected = {
        'model_name': FINAL_MODEL_NAME,
        'model_version': FINAL_MODEL_VERSION,
        'source_experiment_id': reference['experiment_id'],
        'feature_columns': list(FINAL_FEATURE_COLUMNS),
        'threshold': FINAL_THRESHOLD,
        'random_state': FINAL_RANDOM_STATE,
        'training_scope': 'full_train',
    }
    for key, expected_value in expected.items():
        if final_configuration.get(key) != expected_value:
            raise ValueError(f'Frozen final configuration field {key} changed.')
    if final_configuration.get('preprocessing') != reference['preprocessing']:
        raise ValueError('Final preprocessing differs from the reference.')
    if final_configuration.get('model') != reference['model']:
        raise ValueError('Final model parameters differ from the reference.')
    if reference_report.get('experiment_id') != reference['experiment_id']:
        raise ValueError('Reference report experiment identity is invalid.')
    if reference_report.get('evaluation_scope') != 'development_oof_only':
        raise ValueError('Reference report is not development-only.')
    if reference_report.get('lockbox_metrics_computed') is not False:
        raise ValueError('Lockbox metrics were already recorded in selection.')
    expected_hash = final_configuration.get('source_configuration_sha256')
    observed_hash = sha256_canonical_text(reference_configuration_path)
    if expected_hash != observed_hash:
        raise ValueError('Reference configuration hash changed.')
    return final_configuration, reference_report, sha256_canonical_text(final_configuration_path)


def partition_final_training_data(
    training_data: pd.DataFrame,
    assignment: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Return isolated development and lockbox rows after strict checks.'''
    required_training = {'ROW_ID', 'TS', 'class', *FINAL_FEATURE_COLUMNS}
    missing_training = sorted(required_training.difference(training_data.columns))
    if missing_training:
        raise ValueError(f'Training data is missing columns: {missing_training}.')
    required_assignment = {'ROW_ID', 'TS', ROLE_COLUMN, FOLD_COLUMN}
    missing_assignment = sorted(required_assignment.difference(assignment.columns))
    if missing_assignment:
        raise ValueError(f'Assignment is missing columns: {missing_assignment}.')
    if training_data['ROW_ID'].duplicated().any():
        raise ValueError('Training ROW_ID values must be unique.')
    merged = training_data.merge(
        assignment[['ROW_ID', 'TS', ROLE_COLUMN, FOLD_COLUMN]],
        on='ROW_ID', how='left', suffixes=('', '_assignment'),
        validate='one_to_one', sort=False,
    )
    if len(merged) != len(training_data) or merged[ROLE_COLUMN].isna().any():
        raise ValueError('Frozen assignment must cover every training row once.')
    if not merged['TS'].equals(merged['TS_assignment']):
        raise ValueError('Frozen assignment changed at least one TS value.')
    roles = set(merged[ROLE_COLUMN].unique().tolist())
    if roles != {DEVELOPMENT_ROLE, LOCKBOX_ROLE}:
        raise ValueError(f'Frozen assignment roles are invalid: {roles}.')
    development = merged[merged[ROLE_COLUMN] == DEVELOPMENT_ROLE].copy()
    lockbox = merged[merged[ROLE_COLUMN] == LOCKBOX_ROLE].copy()
    if set(development['TS']).intersection(lockbox['TS']):
        raise ValueError('TS groups overlap between development and lockbox.')
    if development[FOLD_COLUMN].isna().any() or lockbox[FOLD_COLUMN].notna().any():
        raise ValueError('Frozen fold identifiers are inconsistent with roles.')
    return development, lockbox


def persist_joblib_atomic(value: Any, path: Path) -> str:
    '''Persist a Joblib artifact atomically and return its SHA-256.'''
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix='.joblib', dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(value, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return file_sha256(path)


def persist_json_atomic(payload: dict[str, Any], path: Path) -> None:
    '''Persist stable UTF-8 JSON atomically.'''
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', newline='\n', suffix='.json',
        dir=path.parent, delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write('\n')
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
