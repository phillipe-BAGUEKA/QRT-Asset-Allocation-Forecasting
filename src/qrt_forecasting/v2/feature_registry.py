'''Deterministic registry for the leakage-safe V2 feature blocks.'''

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final


IDENTIFIER_COLUMNS: Final = ('ROW_ID', 'TS')
TARGET_COLUMNS: Final = ('target', 'TARGET', 'class')
RETURN_RAW_COLUMNS: Final = tuple(f'RET_{index}' for index in range(1, 21))
VOLUME_RAW_COLUMNS: Final = tuple(
    f'SIGNED_VOLUME_{index}' for index in range(1, 21)
)
CATEGORICAL_COLUMNS: Final = ('GROUP', 'ALLOCATION')
WINDOWS: Final = (3, 5, 10, 20)
MINIMUM_OBSERVATIONS: Final = {3: 2, 5: 3, 10: 6, 20: 12}

RETURN_STATISTICS: Final = (
    'MEAN',
    'MEDIAN',
    'COMPOUNDED',
    'VOLATILITY',
    'DOWNSIDE_VOLATILITY',
    'MINIMUM',
    'MAXIMUM',
    'RANGE',
    'Q25',
    'Q75',
    'IQR',
    'POSITIVE_SHARE',
    'SIGN_CHANGES',
    'SLOPE',
    'EWM',
    'MAX_DRAWDOWN',
    'MOMENTUM_VOLATILITY_RATIO',
)
RETURN_DYNAMIC_COLUMNS: Final = tuple(
    f'RET_{statistic}_W{window}'
    for window in WINDOWS
    for statistic in RETURN_STATISTICS
) + (
    'RET_MEAN_SPREAD_W3_W10',
    'RET_MEAN_SPREAD_W5_W20',
    'RET_MEAN_SPREAD_W10_W20',
)
RETURN_ENRICHED_COLUMNS: Final = RETURN_RAW_COLUMNS + RETURN_DYNAMIC_COLUMNS

CROSS_SECTIONAL_ANCHORS: Final = (
    'RET_1',
    'RET_MEAN_W3',
    'RET_MEAN_W5',
    'RET_MEAN_W10',
    'RET_MEAN_W20',
    'RET_COMPOUNDED_W3',
    'RET_COMPOUNDED_W5',
    'RET_COMPOUNDED_W10',
    'RET_COMPOUNDED_W20',
    'RET_VOLATILITY_W5',
    'RET_VOLATILITY_W20',
)
TS_TRANSFORMS: Final = (
    'MEAN',
    'MEDIAN',
    'STD',
    'PERCENTILE',
    'ROBUST_Z',
    'DELTA_MEAN',
    'DELTA_MEDIAN',
)
TS_GROUP_TRANSFORMS: Final = ('DELTA_MEAN', 'PERCENTILE')
CROSS_SECTIONAL_RETURN_COLUMNS: Final = tuple(
    f'{anchor}__TS_{transform}'
    for anchor in CROSS_SECTIONAL_ANCHORS
    for transform in TS_TRANSFORMS
) + tuple(
    f'{anchor}__TS_GROUP_{transform}'
    for anchor in CROSS_SECTIONAL_ANCHORS
    for transform in TS_GROUP_TRANSFORMS
) + tuple(
    f'{anchor}__TS_POSITIVE_SHARE'
    for anchor in (
        'RET_1',
        'RET_COMPOUNDED_W3',
        'RET_COMPOUNDED_W5',
        'RET_COMPOUNDED_W10',
        'RET_COMPOUNDED_W20',
    )
)

VOLUME_AGGREGATE_COLUMNS: Final = (
    'VOLUME_OBSERVED_COUNT',
    'VOLUME_OBSERVED_SHARE',
    'VOLUME_MEAN',
    'VOLUME_MEDIAN',
    'VOLUME_ABS_MEAN',
    'VOLUME_STD',
    'VOLUME_Q25',
    'VOLUME_Q75',
    'VOLUME_IQR',
    'VOLUME_POSITIVE_SHARE',
    'VOLUME_NEGATIVE_SHARE',
    'VOLUME_SHORT_MINUS_LONG',
    'VOLUME_MOST_RECENT_VALUE',
    'VOLUME_MOST_RECENT_LAG',
) + tuple(
    f'{anchor}__TS_{transform}'
    for anchor in (
        'VOLUME_MEAN',
        'VOLUME_ABS_MEAN',
        'VOLUME_STD',
        'VOLUME_OBSERVED_SHARE',
        'VOLUME_SHORT_MINUS_LONG',
    )
    for transform in ('PERCENTILE', 'ROBUST_Z')
)
VOLUME_MISSING_MASK_COLUMNS: Final = tuple(
    f'{column}__MISSING' for column in VOLUME_RAW_COLUMNS
)
VOLUME_RETURN_INTERACTION_COLUMNS: Final = (
    'RETURN_VOLUME_CORRELATION',
    'RETURN_VOLUME_COVARIANCE',
    'RETURN_VOLUME_MEAN_PRODUCT',
    'RETURN_VOLUME_MEAN_ABS_PRODUCT',
    'RETURN_VOLUME_CORRELATION__TS_PERCENTILE',
    'RETURN_VOLUME_CORRELATION__TS_ROBUST_Z',
)
TURNOVER_COLUMNS: Final = (
    'TURNOVER_RAW',
    'TURNOVER_SIGNED_LOG1P',
    'TURNOVER_MISSING',
    'TURNOVER__TS_PERCENTILE',
    'TURNOVER__TS_ROBUST_Z',
    'TURNOVER__TS_GROUP_DELTA_MEAN',
    'TURNOVER_X_RET_MEAN_W3',
    'TURNOVER_X_RET_VOLATILITY_W20',
    'TURNOVER_X_RET_1_TS_DELTA_MEAN',
    'TURNOVER_X_VOLUME_OBSERVED_SHARE',
)


@dataclass(frozen=True)
class FeatureBlock:
    '''One independently cached block with a frozen output order.'''

    name: str
    columns: tuple[str, ...]
    numeric: bool = True
    allows_missing: bool = True


@dataclass(frozen=True)
class FeatureSet:
    '''A declared matrix assembled only from registered blocks.'''

    name: str
    blocks: tuple[str, ...]


FEATURE_BLOCKS: Final = {
    block.name: block
    for block in (
        FeatureBlock('return_raw', RETURN_RAW_COLUMNS),
        FeatureBlock('return_dynamics', RETURN_DYNAMIC_COLUMNS),
        FeatureBlock('return_cross_sectional', CROSS_SECTIONAL_RETURN_COLUMNS),
        FeatureBlock('volume_aggregates', VOLUME_AGGREGATE_COLUMNS),
        FeatureBlock('volume_raw', VOLUME_RAW_COLUMNS),
        FeatureBlock(
            'volume_missing_masks',
            VOLUME_MISSING_MASK_COLUMNS,
            allows_missing=False,
        ),
        FeatureBlock(
            'volume_return_interactions',
            VOLUME_RETURN_INTERACTION_COLUMNS,
        ),
        FeatureBlock('turnover', TURNOVER_COLUMNS),
        FeatureBlock('group', ('GROUP',), numeric=False),
        FeatureBlock('allocation', ('ALLOCATION',), numeric=False),
    )
}

FEATURE_SETS: Final = {
    item.name: item
    for item in (
        FeatureSet('FS0_RET_RAW', ('return_raw',)),
        FeatureSet('FS1_RET_ENRICHED', ('return_raw', 'return_dynamics')),
        FeatureSet(
            'FS2_RET_CROSS_SECTIONAL',
            ('return_raw', 'return_dynamics', 'return_cross_sectional'),
        ),
        FeatureSet(
            'FS3_ALL_NUMERIC_ROBUST',
            (
                'return_raw',
                'return_dynamics',
                'return_cross_sectional',
                'volume_aggregates',
                'turnover',
            ),
        ),
        FeatureSet(
            'FS3_WITH_VOLUME_RAW',
            (
                'return_raw',
                'return_dynamics',
                'return_cross_sectional',
                'volume_aggregates',
                'turnover',
                'volume_raw',
            ),
        ),
        FeatureSet(
            'FS3_WITH_VOLUME_MASKS',
            (
                'return_raw',
                'return_dynamics',
                'return_cross_sectional',
                'volume_aggregates',
                'turnover',
                'volume_missing_masks',
            ),
        ),
        FeatureSet(
            'FS3_WITH_VOLUME_INTERACTIONS',
            (
                'return_raw',
                'return_dynamics',
                'return_cross_sectional',
                'volume_aggregates',
                'turnover',
                'volume_return_interactions',
            ),
        ),
        FeatureSet(
            'FS4_GROUP',
            (
                'return_raw',
                'return_dynamics',
                'return_cross_sectional',
                'volume_aggregates',
                'turnover',
                'group',
            ),
        ),
        FeatureSet(
            'FS5_GROUP_ALLOCATION',
            (
                'return_raw',
                'return_dynamics',
                'return_cross_sectional',
                'volume_aggregates',
                'turnover',
                'group',
                'allocation',
            ),
        ),
    )
}


def feature_columns(feature_set_name: str) -> tuple[str, ...]:
    '''Resolve a feature set from the registry without duplicated lists.'''
    try:
        feature_set = FEATURE_SETS[feature_set_name]
    except KeyError as error:
        raise ValueError(f'Unknown feature set: {feature_set_name}.') from error
    return tuple(
        column
        for block_name in feature_set.blocks
        for column in FEATURE_BLOCKS[block_name].columns
    )


def feature_columns_sha256(columns: tuple[str, ...]) -> str:
    '''Hash a column order using a canonical JSON representation.'''
    payload = json.dumps(list(columns), separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def validate_registry() -> None:
    '''Reject duplicate, forbidden or internally inconsistent declarations.'''
    for feature_set in FEATURE_SETS.values():
        unknown = set(feature_set.blocks).difference(FEATURE_BLOCKS)
        if unknown:
            raise ValueError(
                f'{feature_set.name} references unknown blocks: {sorted(unknown)}.'
            )
        columns = feature_columns(feature_set.name)
        if len(columns) != len(set(columns)):
            raise ValueError(f'{feature_set.name} contains duplicate columns.')
        forbidden = set(columns).intersection(
            {*IDENTIFIER_COLUMNS, *TARGET_COLUMNS}
        )
        if forbidden:
            raise ValueError(
                f'{feature_set.name} contains forbidden columns: {forbidden}.'
            )
        if feature_set.name.startswith(('FS0', 'FS1', 'FS2', 'FS3')):
            categories = set(columns).intersection(CATEGORICAL_COLUMNS)
            if categories:
                raise ValueError(
                    f'{feature_set.name} unexpectedly contains {categories}.'
                )


validate_registry()
