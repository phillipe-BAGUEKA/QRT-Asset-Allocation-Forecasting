from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qrt_forecasting.common.target import (
    build_binary_training_frame,
    create_binary_target,
)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        {'ROW_ID': [3, 1, 2], 'TS': ['C', 'A', 'B'], 'RET_1': [0.3, 0.1, 0.2]}
    )
    target = pd.DataFrame({'ROW_ID': [1, 2, 3], 'target': [-0.1, 0.0, 0.2]})
    return features, target


def test_build_binary_training_frame_preserves_feature_order() -> None:
    features, target = _inputs()

    result = build_binary_training_frame(features, target)

    assert result['ROW_ID'].tolist() == [3, 1, 2]
    assert result['class'].tolist() == [1, 0, 0]
    assert result['target'].tolist() == [0.2, -0.1, 0.0]


@pytest.mark.parametrize(
    'values',
    [
        [0.1, np.nan],
        [0.1, np.inf],
        [0.1, -np.inf],
        ['positive', 'negative'],
        [True, False],
    ],
)
def test_create_binary_target_rejects_invalid_values(values: list[object]) -> None:
    with pytest.raises(ValueError):
        create_binary_target(pd.Series(values))


@pytest.mark.parametrize('frame_name', ['features', 'target'])
def test_duplicate_row_ids_are_rejected(frame_name: str) -> None:
    features, target = _inputs()
    frame = features if frame_name == 'features' else target
    frame.loc[1, 'ROW_ID'] = frame.loc[0, 'ROW_ID']

    with pytest.raises(ValueError, match='duplicate ROW_ID'):
        build_binary_training_frame(features, target)


def test_different_row_id_sets_are_rejected() -> None:
    features, target = _inputs()
    target.loc[0, 'ROW_ID'] = 99

    with pytest.raises(ValueError, match='same ROW_ID'):
        build_binary_training_frame(features, target)
