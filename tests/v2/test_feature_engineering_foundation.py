from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qrt_forecasting.v2.feature_cache import (
    build_development_feature_cache,
    load_cached_feature_set,
)
from qrt_forecasting.v2.feature_engineering import (
    build_feature_blocks,
    wealth_path_drawdowns,
)
from qrt_forecasting.v2.feature_registry import (
    FEATURE_BLOCKS,
    FEATURE_SETS,
    RETURN_RAW_COLUMNS,
    TARGET_COLUMNS,
    feature_columns,
)


def synthetic_raw(n_rows: int = 10) -> pd.DataFrame:
    rows = []
    for row_id in range(n_rows):
        row = {
            'ROW_ID': row_id,
            'TS': row_id // 2,
            'GROUP': f'G{row_id % 2}',
            'ALLOCATION': f'A{row_id % 3}',
            'MEDIAN_DAILY_TURNOVER': float(row_id + 1),
            'target': (-1.0) ** row_id,
        }
        for lag in range(1, 21):
            row[f'RET_{lag}'] = (21 - lag + row_id) / 100.0
            row[f'SIGNED_VOLUME_{lag}'] = float(21 - lag + row_id)
        rows.append(row)
    return pd.DataFrame(rows)


def test_chronological_slope_ewm_and_compounded_return() -> None:
    raw = synthetic_raw(2)
    blocks = build_feature_blocks(raw, partition_name='development')
    dynamics = blocks['return_dynamics']
    assert dynamics.loc[0, 'RET_SLOPE_W20'] == pytest.approx(0.01)
    chronological = raw.loc[0, [f'RET_{i}' for i in range(20, 0, -1)]]
    alpha = 2.0 / 21.0
    expected_ewm = float(chronological.iloc[0])
    for value in chronological.iloc[1:]:
        expected_ewm = alpha * value + (1.0 - alpha) * expected_ewm
    assert dynamics.loc[0, 'RET_EWM_W20'] == pytest.approx(expected_ewm)
    expected_compounded = np.prod(
        1.0 + raw.loc[0, ['RET_3', 'RET_2', 'RET_1']].to_numpy()
    ) - 1.0
    assert dynamics.loc[0, 'RET_COMPOUNDED_W3'] == pytest.approx(
        expected_compounded
    )


def test_corrected_maximum_drawdown_path_is_forty_six_percent() -> None:
    returns = np.array([0.50, -0.40, 0.50, -0.40])
    wealth, peaks, drawdowns = wealth_path_drawdowns(returns)
    np.testing.assert_allclose(wealth, [1.0, 1.5, 0.9, 1.35, 0.81])
    np.testing.assert_allclose(peaks, [1.0, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(drawdowns, [0.0, 0.0, 0.4, 0.1, 0.46])
    assert drawdowns.max() == pytest.approx(0.46)


def test_most_recent_available_volume_respects_lag_order() -> None:
    raw = synthetic_raw(2)
    raw.loc[0, 'SIGNED_VOLUME_1'] = np.nan
    raw.loc[0, 'SIGNED_VOLUME_2'] = 123.0
    aggregates = build_feature_blocks(
        raw, partition_name='development'
    )['volume_aggregates']
    assert aggregates.loc[0, 'VOLUME_MOST_RECENT_VALUE'] == 123.0
    assert aggregates.loc[0, 'VOLUME_MOST_RECENT_LAG'] == 2.0


def test_registry_counts_orders_and_exclusions() -> None:
    expected = {
        'FS0_RET_RAW': 20,
        'FS1_RET_ENRICHED': 91,
        'FS2_RET_CROSS_SECTIONAL': 195,
        'FS3_ALL_NUMERIC_ROBUST': 229,
        'FS3_WITH_VOLUME_RAW': 249,
        'FS3_WITH_VOLUME_MASKS': 249,
        'FS3_WITH_VOLUME_INTERACTIONS': 235,
    }
    assert {
        name: len(feature_columns(name)) for name in expected
    } == expected
    assert feature_columns('FS0_RET_RAW') == RETURN_RAW_COLUMNS
    for feature_set in FEATURE_SETS:
        columns = feature_columns(feature_set)
        assert not {'ROW_ID', 'TS', *TARGET_COLUMNS}.intersection(columns)
    assert 'GROUP' not in feature_columns('FS3_ALL_NUMERIC_ROBUST')
    assert 'ALLOCATION' not in feature_columns('FS3_ALL_NUMERIC_ROBUST')


def test_cross_sectional_features_are_target_free_and_ts_local() -> None:
    raw = synthetic_raw(10)
    first = build_feature_blocks(
        raw, partition_name='development'
    )['return_cross_sectional']
    changed_target = raw.copy()
    changed_target['target'] *= -10_000
    second = build_feature_blocks(
        changed_target, partition_name='development'
    )['return_cross_sectional']
    pd.testing.assert_frame_equal(first, second)

    for ts in raw['TS'].unique():
        subset = raw[raw['TS'] == ts]
        local = build_feature_blocks(
            subset, partition_name='inference'
        )['return_cross_sectional']
        positions = raw.sort_values('ROW_ID').reset_index(drop=True)['TS'].eq(ts)
        pd.testing.assert_frame_equal(
            first.loc[positions].reset_index(drop=True),
            local.reset_index(drop=True),
        )


def test_partition_mixing_is_rejected() -> None:
    raw = synthetic_raw(4)
    raw['role'] = ['development', 'lockbox', 'development', 'lockbox']
    with pytest.raises(ValueError, match='cannot mix partitions'):
        build_feature_blocks(raw, partition_name='development')


def test_feature_engineering_is_invariant_to_source_row_permutation() -> None:
    raw = synthetic_raw(10)
    baseline = build_feature_blocks(raw, partition_name='development')
    shuffled = build_feature_blocks(
        raw.sample(frac=1.0, random_state=42),
        partition_name='development',
    )
    for block_name in FEATURE_BLOCKS:
        pd.testing.assert_frame_equal(
            baseline[block_name], shuffled[block_name]
        )


def test_infinite_inputs_are_rejected_and_masks_are_finite() -> None:
    raw = synthetic_raw(4)
    raw.loc[0, 'SIGNED_VOLUME_1'] = np.nan
    masks = build_feature_blocks(
        raw, partition_name='development'
    )['volume_missing_masks']
    assert np.isfinite(masks.to_numpy()).all()
    raw.loc[0, 'RET_1'] = np.inf
    with pytest.raises(ValueError, match='infinite'):
        build_feature_blocks(raw, partition_name='development')


def _assignment_for_cache() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'ROW_ID': list(range(12)),
            'TS': [index // 2 for index in range(12)],
            'role': ['development'] * 10 + ['lockbox'] * 2,
            'fold_id': pd.array(
                [index // 2 for index in range(10)] + [pd.NA, pd.NA],
                dtype='Int64',
            ),
        }
    )


def test_cache_is_development_only_reproducible_and_hashed(tmp_path) -> None:
    raw = synthetic_raw(12)
    assignment = _assignment_for_cache()
    first_directory = tmp_path / 'first'
    second_directory = tmp_path / 'second'
    first = build_development_feature_cache(
        raw,
        assignment,
        output_directory=first_directory,
        expected_rows=10,
    )
    second = build_development_feature_cache(
        raw.sample(frac=1.0, random_state=9),
        assignment.sample(frac=1.0, random_state=11),
        output_directory=second_directory,
        expected_rows=10,
    )
    assert first['scope'] == 'development_only'
    assert first['n_rows'] == 10
    assert first['lockbox_rows_cached'] == 0
    assert first['test_data_accessed'] is False
    assert first['row_ids_sha256'] == second['row_ids_sha256']
    assert first['folds_sha256'] == second['folds_sha256']
    for block_name in FEATURE_BLOCKS:
        assert (
            first['blocks'][block_name]['content_sha256']
            == second['blocks'][block_name]['content_sha256']
        )
        assert first['blocks'][block_name]['columns_sha256']
    identity, matrix, loaded = load_cached_feature_set(
        first_directory, 'FS3_ALL_NUMERIC_ROBUST'
    )
    assert identity['ROW_ID'].tolist() == list(range(10))
    assert not set(identity['ROW_ID']).intersection({10, 11})
    assert tuple(matrix.columns) == feature_columns(
        'FS3_ALL_NUMERIC_ROBUST'
    )
    assert loaded['duplicate_columns'] == first['duplicate_columns']
    assert loaded['constant_columns'] == first['constant_columns']


def test_cache_refuses_a_lockbox_labelled_as_development(tmp_path) -> None:
    raw = synthetic_raw(12)
    assignment = _assignment_for_cache()
    assignment.loc[10, 'role'] = 'development'
    assignment.loc[10, 'fold_id'] = 0
    with pytest.raises(ValueError):
        build_development_feature_cache(
            raw,
            assignment,
            output_directory=tmp_path / 'cache',
            expected_rows=10,
        )


def test_frozen_v2_assignment_hash_is_unchanged() -> None:
    manifest_path = (
        __import__('pathlib').Path(__file__).resolve().parents[2]
        / 'reports' / 'validation' / 'v2_grouped_folds_manifest.json'
    )
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['assignment_sha256'] == (
        '4c4eaa6133ea4d150b0cfc0efceba9d8686bcb099141f8444b9217f2a211cde7'
    )
