'''Explicit V2 feature families and fold-fitted preprocessing pipelines.'''

from __future__ import annotations

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from qrt_forecasting.v2.gradient_boosting import REFERENCE_MODEL_PARAMETERS


RET_FEATURES = [f'RET_{index}' for index in range(1, 21)]
VOLUME_FEATURES = [f'SIGNED_VOLUME_{index}' for index in range(1, 21)]
TURNOVER_FEATURES = ['MEDIAN_DAILY_TURNOVER']
EXCLUDED_FEATURES = ['ROW_ID', 'TS', 'TARGET']


@dataclass(frozen=True)
class FeatureSet:
    experiment_id: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...] = ()
    add_missing_indicators: bool = False

    @property
    def input_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]


NUMERIC_FEATURE_SETS = (
    FeatureSet('GB_RET_TURNOVER', tuple([*RET_FEATURES, *TURNOVER_FEATURES])),
    FeatureSet('GB_VOLUME_ONLY', tuple(VOLUME_FEATURES)),
    FeatureSet('GB_TURNOVER_ONLY', tuple(TURNOVER_FEATURES)),
    FeatureSet('GB_RET_VOLUME', tuple([*RET_FEATURES, *VOLUME_FEATURES])),
    FeatureSet(
        'GB_RET_VOLUME_INDICATORS',
        tuple([*RET_FEATURES, *VOLUME_FEATURES]),
        add_missing_indicators=True,
    ),
    FeatureSet(
        'GB_ALL_NUMERIC_RAW',
        tuple([*RET_FEATURES, *VOLUME_FEATURES, *TURNOVER_FEATURES]),
    ),
    FeatureSet(
        'GB_ALL_NUMERIC_INDICATORS',
        tuple([*RET_FEATURES, *VOLUME_FEATURES, *TURNOVER_FEATURES]),
        add_missing_indicators=True,
    ),
)


def categorical_feature_sets(best_numeric: FeatureSet) -> tuple[FeatureSet, ...]:
    '''Extend the selected numeric set without changing its preprocessing.'''
    common = {
        'numeric_features': best_numeric.numeric_features,
        'add_missing_indicators': best_numeric.add_missing_indicators,
    }
    return (
        FeatureSet('GB_BEST_NUMERIC_GROUP', categorical_features=('GROUP',), **common),
        FeatureSet(
            'GB_BEST_NUMERIC_ALLOCATION',
            categorical_features=('ALLOCATION',),
            **common,
        ),
        FeatureSet(
            'GB_BEST_NUMERIC_GROUP_ALLOCATION',
            categorical_features=('GROUP', 'ALLOCATION'),
            **common,
        ),
    )


def build_feature_family_pipeline(feature_set: FeatureSet) -> Pipeline:
    '''Create a fresh GB pipeline whose preprocessing is fitted inside a fold.'''
    numeric_imputer = SimpleImputer(
        strategy='constant',
        fill_value=0.0,
        add_indicator=feature_set.add_missing_indicators,
    )
    if feature_set.categorical_features:
        preprocessing = ColumnTransformer(
            transformers=[
                ('numeric', numeric_imputer, list(feature_set.numeric_features)),
                (
                    'categorical',
                    OneHotEncoder(handle_unknown='ignore', sparse_output=True),
                    list(feature_set.categorical_features),
                ),
            ],
            sparse_threshold=1.0,
        )
    else:
        preprocessing = numeric_imputer
    return Pipeline(
        [
            ('preprocessing', preprocessing),
            ('classifier', GradientBoostingClassifier(**REFERENCE_MODEL_PARAMETERS)),
        ]
    )


def classify_feature(column: str) -> str:
    '''Assign one documented feature family to a raw challenge column.'''
    if column == 'ROW_ID':
        return 'identifiants'
    if column == 'TS':
        return 'groupement de validation'
    if column == 'TARGET':
        return 'identifiants'
    if column.startswith('RET_'):
        return 'RET'
    if column.startswith('SIGNED_VOLUME_'):
        return 'SIGNED_VOLUME'
    if column == 'MEDIAN_DAILY_TURNOVER':
        return 'TURNOVER'
    if column == 'GROUP':
        return 'GROUP'
    if column == 'ALLOCATION':
        return 'ALLOCATION'
    return 'autres'
