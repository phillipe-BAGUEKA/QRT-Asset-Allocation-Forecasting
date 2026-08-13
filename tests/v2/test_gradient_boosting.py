from __future__ import annotations

from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

from qrt_forecasting.v2.gradient_boosting import (
    REFERENCE_FEATURE_COLUMNS,
    build_gradient_boosting_reference_pipeline,
    load_experiment_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_reference_pipeline_matches_the_frozen_specification() -> None:
    pipeline = build_gradient_boosting_reference_pipeline()

    assert isinstance(pipeline.named_steps['imputer'], SimpleImputer)
    assert pipeline.named_steps['imputer'].strategy == 'constant'
    assert pipeline.named_steps['imputer'].fill_value == 0.0
    classifier = pipeline.named_steps['classifier']
    assert isinstance(classifier, GradientBoostingClassifier)
    assert classifier.learning_rate == 0.05
    assert classifier.n_estimators == 50
    assert classifier.max_depth == 2
    assert classifier.min_samples_leaf == 20
    assert classifier.subsample == 0.7
    assert classifier.max_features == 'sqrt'
    assert classifier.random_state == 42


def test_pipeline_factory_returns_fresh_unfitted_instances() -> None:
    first = build_gradient_boosting_reference_pipeline()
    second = build_gradient_boosting_reference_pipeline()

    assert first is not second
    assert first.named_steps['classifier'] is not second.named_steps['classifier']
    assert not hasattr(first.named_steps['classifier'], 'estimators_')


def test_reference_configuration_is_exact() -> None:
    configuration = load_experiment_configuration(
        REPOSITORY_ROOT / 'configs' / 'experiments' / 'GB_RET20_REFERENCE.toml'
    )

    assert configuration['feature_columns'] == REFERENCE_FEATURE_COLUMNS
    assert configuration['threshold'] == 0.5
    assert configuration['random_state'] == 42
    assert configuration['validation']['n_splits'] == 5
