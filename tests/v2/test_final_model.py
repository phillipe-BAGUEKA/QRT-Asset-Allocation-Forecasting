from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

from qrt_forecasting.v2.final_model import (
    FINAL_FEATURE_COLUMNS,
    FINAL_MODEL_NAME,
    FINAL_MODEL_VERSION,
    FINAL_THRESHOLD,
    build_final_gradient_boosting_pipeline,
    partition_final_training_data,
    validate_frozen_final_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _training_data() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            'ROW_ID': [1, 2, 3, 4],
            'TS': ['A', 'A', 'B', 'B'],
            'class': [0, 1, 0, 1],
        }
    )
    for index, column in enumerate(FINAL_FEATURE_COLUMNS):
        frame[column] = np.asarray([-1.0, 1.0, -0.5, 0.5]) + index
    return frame


def _assignment() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'ROW_ID': [1, 2, 3, 4],
            'TS': ['A', 'A', 'B', 'B'],
            'role': ['development', 'development', 'lockbox', 'lockbox'],
            'fold_id': pd.array([0, 0, None, None], dtype='Int64'),
        }
    )


def test_final_pipeline_contains_frozen_ordered_steps_and_parameters() -> None:
    first = build_final_gradient_boosting_pipeline()
    second = build_final_gradient_boosting_pipeline()

    assert first is not second
    assert list(first.named_steps) == ['feature_selection', 'imputer', 'classifier']
    assert tuple(first.named_steps['feature_selection'].columns) == FINAL_FEATURE_COLUMNS
    assert isinstance(first.named_steps['imputer'], SimpleImputer)
    assert first.named_steps['imputer'].strategy == 'constant'
    assert first.named_steps['imputer'].fill_value == 0.0
    classifier = first.named_steps['classifier']
    assert isinstance(classifier, GradientBoostingClassifier)
    assert classifier.get_params()['n_estimators'] == 50
    assert classifier.get_params()['learning_rate'] == 0.05
    assert classifier.get_params()['max_depth'] == 2
    assert classifier.get_params()['min_samples_leaf'] == 20
    assert classifier.get_params()['subsample'] == 0.7
    assert classifier.get_params()['max_features'] == 'sqrt'
    assert classifier.get_params()['random_state'] == 42
    assert FINAL_THRESHOLD == 0.5


def test_selector_enforces_schema_and_order() -> None:
    training = _training_data()
    reversed_features = list(reversed(FINAL_FEATURE_COLUMNS))
    pipeline = build_final_gradient_boosting_pipeline()
    pipeline.fit(training[reversed_features], training['class'])

    selected = pipeline.named_steps['feature_selection'].transform(
        training[reversed_features]
    )
    assert selected.columns.tolist() == list(FINAL_FEATURE_COLUMNS)
    with pytest.raises(ValueError, match='missing columns'):
        pipeline.predict_proba(training[list(FINAL_FEATURE_COLUMNS[:-1])])


def test_frozen_final_configuration_matches_reference_and_is_development_only() -> None:
    configuration, report, digest = validate_frozen_final_configuration(
        REPOSITORY_ROOT / 'configs' / 'final' / 'GRADIENT_BOOSTING_RET20_FINAL.toml',
        REPOSITORY_ROOT / 'configs' / 'experiments' / 'GB_RET20_REFERENCE.toml',
        REPOSITORY_ROOT / 'reports' / 'experiments' / 'GB_RET20_REFERENCE.json',
    )
    assert configuration['model_name'] == FINAL_MODEL_NAME
    assert configuration['model_version'] == FINAL_MODEL_VERSION
    assert report['lockbox_metrics_computed'] is False
    assert report['evaluation_scope'] == 'development_oof_only'
    assert len(digest) == 64


def test_partition_preserves_group_isolation_and_frozen_roles() -> None:
    development, lockbox = partition_final_training_data(
        _training_data(), _assignment()
    )
    assert development['ROW_ID'].tolist() == [1, 2]
    assert lockbox['ROW_ID'].tolist() == [3, 4]
    assert set(development['TS']).isdisjoint(lockbox['TS'])


def test_partition_rejects_cross_role_group() -> None:
    assignment = _assignment()
    assignment.loc[2, 'TS'] = 'A'
    training = _training_data()
    training.loc[2, 'TS'] = 'A'
    with pytest.raises(ValueError, match='overlap'):
        partition_final_training_data(training, assignment)
