from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from qrt_forecasting.v2.feature_families import (
    EXCLUDED_FEATURES,
    NUMERIC_FEATURE_SETS,
    RET_FEATURES,
    TURNOVER_FEATURES,
    VOLUME_FEATURES,
    FeatureSet,
    build_feature_family_pipeline,
    categorical_feature_sets,
    classify_feature,
)
from qrt_forecasting.v2.evaluation import evaluate_grouped_oof


def test_feature_family_order_and_exclusions_are_explicit() -> None:
    assert RET_FEATURES == [f'RET_{index}' for index in range(1, 21)]
    assert VOLUME_FEATURES == [
        f'SIGNED_VOLUME_{index}' for index in range(1, 21)
    ]
    assert TURNOVER_FEATURES == ['MEDIAN_DAILY_TURNOVER']
    assert EXCLUDED_FEATURES == ['ROW_ID', 'TS', 'TARGET']
    assert [item.experiment_id for item in NUMERIC_FEATURE_SETS] == [
        'GB_RET_TURNOVER',
        'GB_VOLUME_ONLY',
        'GB_TURNOVER_ONLY',
        'GB_RET_VOLUME',
        'GB_RET_VOLUME_INDICATORS',
        'GB_ALL_NUMERIC_RAW',
        'GB_ALL_NUMERIC_INDICATORS',
    ]


def test_feature_classification_covers_required_families() -> None:
    assert classify_feature('ROW_ID') == 'identifiants'
    assert classify_feature('TS') == 'groupement de validation'
    assert classify_feature('RET_1') == 'RET'
    assert classify_feature('SIGNED_VOLUME_1') == 'SIGNED_VOLUME'
    assert classify_feature('MEDIAN_DAILY_TURNOVER') == 'TURNOVER'
    assert classify_feature('GROUP') == 'GROUP'
    assert classify_feature('ALLOCATION') == 'ALLOCATION'
    assert classify_feature('UNKNOWN') == 'autres'


def test_missing_indicators_are_learned_by_the_training_imputer() -> None:
    feature_set = FeatureSet(
        'TEST', ('RET_1', 'RET_2'), add_missing_indicators=True
    )
    pipeline = build_feature_family_pipeline(feature_set)
    frame = pd.DataFrame({'RET_1': [1.0, np.nan, 2.0, 3.0], 'RET_2': [1.0] * 4})
    pipeline.fit(frame, [0, 1, 0, 1])
    imputer = pipeline.named_steps['preprocessing']

    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == 'constant'
    assert imputer.fill_value == 0.0
    assert imputer.add_indicator is True
    assert imputer.indicator_.features_.tolist() == [0]


def test_categorical_encoding_is_one_hot_and_fold_fitted() -> None:
    best = FeatureSet('BEST', ('RET_1',))
    feature_set = categorical_feature_sets(best)[2]
    pipeline = build_feature_family_pipeline(feature_set)
    frame = pd.DataFrame(
        {
            'RET_1': [0.0, 1.0, 0.1, 0.9],
            'GROUP': [1, 2, 1, 2],
            'ALLOCATION': ['A', 'B', 'A', 'B'],
        }
    )
    pipeline.fit(frame, [0, 1, 0, 1])
    preprocessing = pipeline.named_steps['preprocessing']
    encoder = preprocessing.named_transformers_['categorical']

    assert isinstance(preprocessing, ColumnTransformer)
    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == 'ignore'
    assert [values.tolist() for values in encoder.categories_] == [
        [1, 2],
        ['A', 'B'],
    ]


def test_missing_indicators_are_fitted_separately_inside_each_fold() -> None:
    rows = []
    assignments = []
    for group in range(10):
        fold = group % 5
        for offset in range(4):
            row_id = group * 4 + offset
            rows.append(
                {
                    'ROW_ID': row_id,
                    'TS': f'TS_{group}',
                    'class': offset % 2,
                    'RET_1': np.nan if fold == 0 else float(offset),
                }
            )
            assignments.append(
                {
                    'ROW_ID': row_id,
                    'TS': f'TS_{group}',
                    'role': 'development',
                    'fold_id': fold,
                }
            )
    pipelines = []
    feature_set = FeatureSet('TEST', ('RET_1',), add_missing_indicators=True)

    def factory():
        pipeline = build_feature_family_pipeline(feature_set)
        pipelines.append(pipeline)
        return pipeline

    evaluate_grouped_oof(
        pd.DataFrame(rows),
        pd.DataFrame(assignments),
        feature_columns=['RET_1'],
        pipeline_factory=factory,
    )

    assert pipelines[0].named_steps['preprocessing'].indicator_.features_.size == 0
    for pipeline in pipelines[1:]:
        assert pipeline.named_steps[
            'preprocessing'
        ].indicator_.features_.tolist() == [0]
