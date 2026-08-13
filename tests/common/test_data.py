from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qrt_forecasting.common.data import (
    DEFAULT_RAW_DATA_DIRECTORY,
    PROJECT_ROOT,
    load_training_features,
    load_training_target,
)


def test_default_paths_are_derived_from_module_location() -> None:
    assert PROJECT_ROOT == Path(__file__).resolve().parents[2]
    assert DEFAULT_RAW_DATA_DIRECTORY == PROJECT_ROOT / 'data' / 'raw'


def test_training_loaders_support_explicit_directory_and_columns(
    tmp_path: Path,
) -> None:
    pd.DataFrame({'ROW_ID': [1], 'TS': ['GROUP_A']}).to_csv(
        tmp_path / 'X_train.csv', index=False
    )
    pd.DataFrame({'ROW_ID': [1], 'target': [0.1]}).to_csv(
        tmp_path / 'y_train.csv', index=False
    )

    features = load_training_features(tmp_path, usecols=['ROW_ID', 'TS'])
    target = load_training_target(tmp_path, usecols=['ROW_ID', 'target'])

    assert features.to_dict('records') == [{'ROW_ID': 1, 'TS': 'GROUP_A'}]
    assert target.to_dict('records') == [{'ROW_ID': 1, 'target': 0.1}]


def test_missing_training_file_raises_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='X_train.csv'):
        load_training_features(tmp_path)
