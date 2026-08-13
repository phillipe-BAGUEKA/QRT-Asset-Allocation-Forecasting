'''Validated construction of the binary QRT classification target.'''

from __future__ import annotations

import numpy as np
import pandas as pd


ROW_ID_COLUMN = 'ROW_ID'
RAW_TARGET_COLUMN = 'target'
BINARY_TARGET_COLUMN = 'class'


def _validate_row_ids(frame: pd.DataFrame, frame_name: str) -> None:
    if ROW_ID_COLUMN not in frame.columns:
        raise ValueError(f'{frame_name} is missing required column ROW_ID.')
    if frame[ROW_ID_COLUMN].isna().any():
        raise ValueError(f'{frame_name} contains missing ROW_ID values.')
    if frame[ROW_ID_COLUMN].duplicated().any():
        raise ValueError(f'{frame_name} contains duplicate ROW_ID values.')


def create_binary_target(target: pd.Series) -> pd.Series:
    '''Return 1 for finite positive targets and 0 otherwise.'''
    if target.isna().any():
        raise ValueError('target contains missing values.')
    if pd.api.types.is_bool_dtype(target.dtype):
        raise ValueError('target must be numeric and non-boolean.')
    if not pd.api.types.is_numeric_dtype(target.dtype):
        raise ValueError('target must be numeric and non-boolean.')
    values = target.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('target must contain only finite values.')
    return pd.Series(
        (values > 0.0).astype('int8'),
        index=target.index,
        name=BINARY_TARGET_COLUMN,
    )


def build_binary_training_frame(
    training_features: pd.DataFrame,
    training_target: pd.DataFrame,
) -> pd.DataFrame:
    '''Merge validated training inputs and append the binary class.'''
    if not isinstance(training_features, pd.DataFrame):
        raise TypeError('training_features must be a pandas DataFrame.')
    if not isinstance(training_target, pd.DataFrame):
        raise TypeError('training_target must be a pandas DataFrame.')

    _validate_row_ids(training_features, 'training_features')
    _validate_row_ids(training_target, 'training_target')
    if RAW_TARGET_COLUMN not in training_target.columns:
        raise ValueError('training_target is missing required column target.')

    feature_ids = set(training_features[ROW_ID_COLUMN].tolist())
    target_ids = set(training_target[ROW_ID_COLUMN].tolist())
    if feature_ids != target_ids:
        raise ValueError(
            'training_features and training_target must contain exactly '
            'the same ROW_ID values.'
        )

    target_frame = training_target[[ROW_ID_COLUMN, RAW_TARGET_COLUMN]].copy()
    target_frame[BINARY_TARGET_COLUMN] = create_binary_target(
        target_frame[RAW_TARGET_COLUMN]
    )
    merged = training_features.merge(
        target_frame,
        on=ROW_ID_COLUMN,
        how='left',
        sort=False,
        validate='one_to_one',
    )
    if len(merged) != len(training_features):
        raise ValueError('Target merge changed the training row count.')
    return merged
