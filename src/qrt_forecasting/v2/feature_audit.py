'''Reproducible development feature inventory and category audit.'''

from __future__ import annotations

from typing import Any

import pandas as pd

from qrt_forecasting.v2.feature_families import classify_feature


def _categorical_row(
    column: str,
    development: set[str],
    test: set[str],
    dense_bytes: int,
    sparse_bytes: int,
) -> dict[str, Any]:
    return {
        'column': column,
        'development_cardinality': len(development),
        'test_cardinality': len(test),
        'common_count': len(development & test),
        'development_only_count': len(development - test),
        'test_only_count': len(test - development),
        'development_only_categories': sorted(development - test),
        'test_only_categories': sorted(test - development),
        'estimated_dense_float64_megabytes_per_fold': dense_bytes / 1_000_000,
        'estimated_sparse_csr_megabytes_per_fold': sparse_bytes / 1_000_000,
        'one_hot_fold_safe_reasonable': len(development) <= 500
        and sparse_bytes < 512_000_000,
        'proposed_encoding': (
            'OneHotEncoder avec handle_unknown=ignore, ajusté dans chaque fold'
        ),
    }


def audit_feature_columns(
    development_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    '''Describe raw columns on development without consulting any target.'''
    rows: list[dict[str, Any]] = []
    columns = list(
        dict.fromkeys([*development_features.columns, *test_features.columns])
    )
    for column in columns:
        series = development_features[column]
        numeric = pd.api.types.is_numeric_dtype(series.dtype)
        row: dict[str, Any] = {
            'name': column,
            'family': classify_feature(column),
            'dtype': str(series.dtype),
            'n_unique': int(series.nunique(dropna=True)),
            'missing_rate': float(series.isna().mean()),
            'in_train': column in development_features.columns,
            'in_test': column in test_features.columns,
            'minimum': None,
            'q05': None,
            'q25': None,
            'mean': None,
            'median': None,
            'q75': None,
            'q95': None,
            'maximum': None,
            'cardinality': int(series.nunique(dropna=True)),
        }
        if numeric and not series.dropna().empty:
            quantiles = series.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            row.update(
                {
                    'minimum': float(series.min()),
                    'q05': float(quantiles.loc[0.05]),
                    'q25': float(quantiles.loc[0.25]),
                    'mean': float(series.mean()),
                    'median': float(quantiles.loc[0.5]),
                    'q75': float(quantiles.loc[0.75]),
                    'q95': float(quantiles.loc[0.95]),
                    'maximum': float(series.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def audit_categorical_overlap(
    development_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    columns: tuple[str, ...] = ('GROUP', 'ALLOCATION'),
    numeric_feature_count: int = 41,
) -> list[dict[str, Any]]:
    '''Compare categories and estimate dense and sparse one-hot memory costs.'''
    n_fold_train = int(round(len(development_features) * 4 / 5))
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in development_features or column not in test_features:
            raise ValueError(f'Missing categorical audit column {column}.')
        development = set(development_features[column].dropna().astype(str))
        test = set(test_features[column].dropna().astype(str))
        cardinality = len(development)
        dense_bytes = n_fold_train * (numeric_feature_count + cardinality) * 8
        sparse_bytes = (
            n_fold_train * (numeric_feature_count + 1) * 12
            + (n_fold_train + 1) * 4
        )
        rows.append(
            _categorical_row(
                column, development, test, dense_bytes, sparse_bytes
            )
        )
    return rows
