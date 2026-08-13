'''Repository-relative access to the labelled QRT training data.'''

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DATA_DIRECTORY = PROJECT_ROOT / 'data' / 'raw'


def _resolve_training_file(
    filename: str,
    data_directory: str | Path | None,
) -> Path:
    directory = (
        DEFAULT_RAW_DATA_DIRECTORY
        if data_directory is None
        else Path(data_directory).expanduser().resolve()
    )
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f'Required QRT training file not found: {path}')
    return path


def load_training_features(
    data_directory: str | Path | None = None,
    *,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    '''Load X_train.csv without accessing the challenge test data.'''
    path = _resolve_training_file('X_train.csv', data_directory)
    return pd.read_csv(path, usecols=usecols)


def load_training_target(
    data_directory: str | Path | None = None,
    *,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    '''Load y_train.csv without accessing the challenge test data.'''
    path = _resolve_training_file('y_train.csv', data_directory)
    return pd.read_csv(path, usecols=usecols)
