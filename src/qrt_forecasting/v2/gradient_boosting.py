'''Frozen Gradient Boosting reference configuration for V2.'''

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


REFERENCE_EXPERIMENT_ID = 'GB_RET20_REFERENCE'
REFERENCE_FEATURE_COLUMNS = [f'RET_{index}' for index in range(1, 21)]
REFERENCE_THRESHOLD = 0.5
REFERENCE_RANDOM_STATE = 42
REFERENCE_MODEL_PARAMETERS: dict[str, Any] = {
    'learning_rate': 0.05,
    'n_estimators': 50,
    'max_depth': 2,
    'min_samples_leaf': 20,
    'subsample': 0.7,
    'max_features': 'sqrt',
    'random_state': REFERENCE_RANDOM_STATE,
}


def build_gradient_boosting_reference_pipeline() -> Pipeline:
    '''Return a new, unfitted instance of the frozen V2 reference pipeline.'''
    return Pipeline(
        steps=[
            (
                'imputer',
                SimpleImputer(strategy='constant', fill_value=0.0),
            ),
            (
                'classifier',
                GradientBoostingClassifier(**REFERENCE_MODEL_PARAMETERS),
            ),
        ]
    )


def load_experiment_configuration(path: Path) -> dict[str, Any]:
    '''Load and validate the immutable reference experiment declaration.'''
    path = Path(path).resolve()
    with path.open('rb') as configuration_file:
        configuration = tomllib.load(configuration_file)

    expected = {
        'experiment_id': REFERENCE_EXPERIMENT_ID,
        'feature_columns': REFERENCE_FEATURE_COLUMNS,
        'target_column': 'class',
        'group_column': 'TS',
        'row_id_column': 'ROW_ID',
        'threshold': REFERENCE_THRESHOLD,
        'random_state': REFERENCE_RANDOM_STATE,
    }
    for key, expected_value in expected.items():
        if configuration.get(key) != expected_value:
            raise ValueError(
                f'Invalid reference configuration for {key}: '
                f'{configuration.get(key)!r}.'
            )
    if configuration.get('preprocessing') != {
        'strategy': 'constant',
        'fill_value': 0.0,
    }:
        raise ValueError('The reference preprocessing configuration changed.')
    expected_model = dict(REFERENCE_MODEL_PARAMETERS)
    expected_model.pop('random_state')
    if configuration.get('model') != expected_model:
        raise ValueError('The reference Gradient Boosting parameters changed.')
    if configuration.get('validation', {}).get('n_splits') != 5:
        raise ValueError('The reference experiment requires exactly five folds.')
    return configuration


def file_sha256(path: Path) -> str:
    '''Return the SHA-256 digest of one file without transforming its bytes.'''
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_canonical_text(path: Path) -> str:
    '''Hash methodological text after canonicalizing CRLF line endings to LF.'''
    raw_content = Path(path).read_bytes()
    canonical_content = raw_content.replace(b'\r\n', b'\n')
    return hashlib.sha256(canonical_content).hexdigest()
