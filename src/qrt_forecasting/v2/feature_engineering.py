'''Target-free, deterministic V2 feature engineering.'''

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from qrt_forecasting.v2.feature_registry import (
    CROSS_SECTIONAL_ANCHORS,
    CROSS_SECTIONAL_RETURN_COLUMNS,
    FEATURE_BLOCKS,
    MINIMUM_OBSERVATIONS,
    RETURN_DYNAMIC_COLUMNS,
    RETURN_RAW_COLUMNS,
    RETURN_STATISTICS,
    TURNOVER_COLUMNS,
    VOLUME_AGGREGATE_COLUMNS,
    VOLUME_MISSING_MASK_COLUMNS,
    VOLUME_RAW_COLUMNS,
    VOLUME_RETURN_INTERACTION_COLUMNS,
    WINDOWS,
)

EPSILON = 1e-12
ALLOWED_PARTITIONS = frozenset({'development', 'lockbox', 'test', 'inference'})


def wealth_path_drawdowns(
    chronological_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''Return wealth, running peaks and drawdowns from initial wealth one.'''
    values = np.asarray(chronological_returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError('chronological_returns must be one-dimensional.')
    finite = values[np.isfinite(values)]
    if np.isinf(values).any() or (finite <= -1.0).any():
        raise ValueError('Returns must be finite-or-missing and above -1.')
    wealth = np.concatenate(
        ([1.0], np.cumprod(1.0 + np.nan_to_num(values, nan=0.0)))
    )
    peaks = np.maximum.accumulate(wealth)
    return wealth, peaks, 1.0 - wealth / peaks


def _validate_source(frame: pd.DataFrame, partition_name: str) -> pd.DataFrame:
    if partition_name not in ALLOWED_PARTITIONS:
        raise ValueError(
            f'partition_name must be one of {sorted(ALLOWED_PARTITIONS)}.'
        )
    required = {
        'ROW_ID', 'TS', 'GROUP', 'ALLOCATION', 'MEDIAN_DAILY_TURNOVER',
        *RETURN_RAW_COLUMNS, *VOLUME_RAW_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f'Missing feature-engineering columns: {missing}.')
    if frame.empty:
        raise ValueError('Feature engineering requires at least one row.')
    if frame['ROW_ID'].isna().any() or frame['ROW_ID'].duplicated().any():
        raise ValueError('ROW_ID values must be present and unique.')
    if frame['TS'].isna().any():
        raise ValueError('TS values must be present.')
    if 'role' in frame.columns:
        roles = set(frame['role'].dropna().astype(str).unique().tolist())
        if roles != {partition_name}:
            raise ValueError('Cross-sectional features cannot mix partitions.')
        if frame.groupby('TS', observed=True)['role'].nunique().gt(1).any():
            raise ValueError('A TS group crosses multiple data partitions.')
    numeric_columns = [
        *RETURN_RAW_COLUMNS, *VOLUME_RAW_COLUMNS, 'MEDIAN_DAILY_TURNOVER'
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors='raise')
    if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
        raise ValueError('Raw feature inputs contain infinite values.')
    return frame.sort_values('ROW_ID', kind='mergesort').reset_index(drop=True)


def _ewm_last(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    state = np.full(values.shape[0], np.nan)
    seen = np.zeros(values.shape[0], dtype=bool)
    for current in values.T:
        valid = np.isfinite(current)
        state = np.where(
            valid & seen,
            alpha * current + (1.0 - alpha) * state,
            np.where(valid, current, state),
        )
        seen |= valid
    return state


def _slope(values: np.ndarray) -> np.ndarray:
    positions = np.arange(values.shape[1], dtype=np.float64)
    valid = np.isfinite(values)
    count = valid.sum(axis=1).astype(float)
    x_sum = (valid * positions).sum(axis=1)
    y_sum = np.nansum(values, axis=1)
    xy_sum = np.nansum(values * positions, axis=1)
    x2_sum = (valid * positions**2).sum(axis=1)
    denominator = count * x2_sum - x_sum**2
    return np.divide(
        count * xy_sum - x_sum * y_sum,
        denominator,
        out=np.full(values.shape[0], np.nan),
        where=np.abs(denominator) > EPSILON,
    )


def _max_drawdown(values: np.ndarray) -> np.ndarray:
    wealth = np.cumprod(1.0 + np.nan_to_num(values, nan=0.0), axis=1)
    wealth = np.concatenate([np.ones((len(values), 1)), wealth], axis=1)
    peaks = np.maximum.accumulate(wealth, axis=1)
    return np.max(1.0 - wealth / peaks, axis=1)


def _window_statistics(values: np.ndarray, window: int) -> dict[str, np.ndarray]:
    counts = np.isfinite(values).sum(axis=1)
    enough = counts >= MINIMUM_OBSERVATIONS[window]
    with np.errstate(all='ignore'):
        mean = np.nanmean(values, axis=1)
        median = np.nanmedian(values, axis=1)
        volatility = np.nanstd(values, axis=1, ddof=0)
        minimum = np.nanmin(values, axis=1)
        maximum = np.nanmax(values, axis=1)
        q25 = np.nanquantile(values, 0.25, axis=1)
        q75 = np.nanquantile(values, 0.75, axis=1)
        downside = np.sqrt(np.nanmean(np.minimum(values, 0.0) ** 2, axis=1))
        compounded = np.expm1(np.nansum(np.log1p(values), axis=1))
    pairs = np.isfinite(values[:, :-1]) & np.isfinite(values[:, 1:])
    sign_changes = (
        pairs & ((values[:, :-1] >= 0.0) != (values[:, 1:] >= 0.0))
    ).sum(axis=1).astype(float)
    positive_share = np.divide(
        (values > 0.0).sum(axis=1), counts,
        out=np.full(len(values), np.nan), where=counts > 0
    )
    ratio = np.divide(
        mean, volatility, out=np.full(len(values), np.nan),
        where=volatility > EPSILON
    )
    output = {
        'MEAN': mean, 'MEDIAN': median, 'COMPOUNDED': compounded,
        'VOLATILITY': volatility, 'DOWNSIDE_VOLATILITY': downside,
        'MINIMUM': minimum, 'MAXIMUM': maximum, 'RANGE': maximum - minimum,
        'Q25': q25, 'Q75': q75, 'IQR': q75 - q25,
        'POSITIVE_SHARE': positive_share, 'SIGN_CHANGES': sign_changes,
        'SLOPE': _slope(values), 'EWM': _ewm_last(values, window),
        'MAX_DRAWDOWN': _max_drawdown(values),
        'MOMENTUM_VOLATILITY_RATIO': ratio,
    }
    return {
        statistic: np.where(enough, output[statistic], np.nan)
        for statistic in RETURN_STATISTICS
    }


def build_return_blocks(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Build raw and enriched return blocks in frozen column order.'''
    raw = frame[list(RETURN_RAW_COLUMNS)].astype('float32')
    generated: dict[str, np.ndarray] = {}
    for window in WINDOWS:
        columns = [f'RET_{index}' for index in range(window, 0, -1)]
        values = frame[columns].to_numpy(dtype=np.float64)
        for statistic, result in _window_statistics(values, window).items():
            generated[f'RET_{statistic}_W{window}'] = result
    generated['RET_MEAN_SPREAD_W3_W10'] = (
        generated['RET_MEAN_W3'] - generated['RET_MEAN_W10']
    )
    generated['RET_MEAN_SPREAD_W5_W20'] = (
        generated['RET_MEAN_W5'] - generated['RET_MEAN_W20']
    )
    generated['RET_MEAN_SPREAD_W10_W20'] = (
        generated['RET_MEAN_W10'] - generated['RET_MEAN_W20']
    )
    dynamics = pd.DataFrame(generated)[list(RETURN_DYNAMIC_COLUMNS)]
    return raw, dynamics.astype('float32')


def _robust_z(values: pd.Series, groups: pd.Series) -> pd.Series:
    median = values.groupby(groups, observed=True).transform('median')
    mad = (values - median).abs().groupby(groups, observed=True).transform(
        'median'
    )
    return (values - median) / (1.4826 * mad).where(mad > EPSILON)


def _percentile(values: pd.Series, groups: pd.Series) -> pd.Series:
    return values.groupby(groups, observed=True).rank(
        method='average', pct=True, na_option='keep'
    )


def _subgroups(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(zip(frame['TS'].tolist(), frame['GROUP'].tolist())),
        index=frame.index,
        dtype='object',
    )


def _cross_sectional_return_block(
    frame: pd.DataFrame,
    returns: pd.DataFrame,
) -> pd.DataFrame:
    groups = frame['TS']
    subgroups = _subgroups(frame)
    output: dict[str, pd.Series] = {}
    for anchor in CROSS_SECTIONAL_ANCHORS:
        values = returns[anchor]
        mean = values.groupby(groups, observed=True).transform('mean')
        median = values.groupby(groups, observed=True).transform('median')
        output[f'{anchor}__TS_MEAN'] = mean
        output[f'{anchor}__TS_MEDIAN'] = median
        output[f'{anchor}__TS_STD'] = values.groupby(
            groups, observed=True
        ).transform(lambda item: item.std(ddof=0))
        output[f'{anchor}__TS_PERCENTILE'] = _percentile(values, groups)
        output[f'{anchor}__TS_ROBUST_Z'] = _robust_z(values, groups)
        output[f'{anchor}__TS_DELTA_MEAN'] = values - mean
        output[f'{anchor}__TS_DELTA_MEDIAN'] = values - median
        subgroup_mean = values.groupby(
            subgroups, observed=True
        ).transform('mean')
        output[f'{anchor}__TS_GROUP_DELTA_MEAN'] = values - subgroup_mean
        output[f'{anchor}__TS_GROUP_PERCENTILE'] = _percentile(
            values, subgroups
        )
    positive_anchors = (
        'RET_1', 'RET_COMPOUNDED_W3', 'RET_COMPOUNDED_W5',
        'RET_COMPOUNDED_W10', 'RET_COMPOUNDED_W20',
    )
    for anchor in positive_anchors:
        values = returns[anchor]
        output[f'{anchor}__TS_POSITIVE_SHARE'] = (
            values.gt(0.0).where(values.notna())
            .groupby(groups, observed=True).transform('mean')
        )
    return pd.DataFrame(output)[
        list(CROSS_SECTIONAL_RETURN_COLUMNS)
    ].astype('float32')


def _volume_aggregates(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = frame[list(VOLUME_RAW_COLUMNS)].astype('float32')
    values = raw.to_numpy(dtype=np.float64)
    observed = np.isfinite(values)
    counts = observed.sum(axis=1)
    with np.errstate(all='ignore'):
        mean = np.nanmean(values, axis=1)
        median = np.nanmedian(values, axis=1)
        abs_mean = np.nanmean(np.abs(values), axis=1)
        std = np.nanstd(values, axis=1, ddof=0)
        q25 = np.nanquantile(values, 0.25, axis=1)
        q75 = np.nanquantile(values, 0.75, axis=1)
        short_mean = np.nanmean(values[:, :5], axis=1)
    first = np.argmax(observed, axis=1)
    any_value = counts > 0
    most_recent = np.where(
        any_value, values[np.arange(len(values)), first], np.nan
    )
    aggregates = pd.DataFrame(
        {
            'VOLUME_OBSERVED_COUNT': counts,
            'VOLUME_OBSERVED_SHARE': counts / 20.0,
            'VOLUME_MEAN': mean,
            'VOLUME_MEDIAN': median,
            'VOLUME_ABS_MEAN': abs_mean,
            'VOLUME_STD': std,
            'VOLUME_Q25': q25,
            'VOLUME_Q75': q75,
            'VOLUME_IQR': q75 - q25,
            'VOLUME_POSITIVE_SHARE': np.divide(
                (values > 0.0).sum(axis=1), counts,
                out=np.full(len(values), np.nan), where=counts > 0
            ),
            'VOLUME_NEGATIVE_SHARE': np.divide(
                (values < 0.0).sum(axis=1), counts,
                out=np.full(len(values), np.nan), where=counts > 0
            ),
            'VOLUME_SHORT_MINUS_LONG': np.where(
                (np.isfinite(values[:, :5]).sum(axis=1) >= 3)
                & (counts >= 12),
                short_mean - mean,
                np.nan,
            ),
            'VOLUME_MOST_RECENT_VALUE': most_recent,
            'VOLUME_MOST_RECENT_LAG': np.where(
                any_value, first + 1, np.nan
            ),
        }
    )
    for anchor in (
        'VOLUME_MEAN', 'VOLUME_ABS_MEAN', 'VOLUME_STD',
        'VOLUME_OBSERVED_SHARE', 'VOLUME_SHORT_MINUS_LONG',
    ):
        aggregates[f'{anchor}__TS_PERCENTILE'] = _percentile(
            aggregates[anchor], frame['TS']
        )
        aggregates[f'{anchor}__TS_ROBUST_Z'] = _robust_z(
            aggregates[anchor], frame['TS']
        )
    masks = raw.isna().astype('float32')
    masks.columns = list(VOLUME_MISSING_MASK_COLUMNS)
    return (
        aggregates[list(VOLUME_AGGREGATE_COLUMNS)].astype('float32'),
        raw,
        masks,
    )


def _volume_return_interactions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    returns = frame[
        [f'RET_{index}' for index in range(20, 0, -1)]
    ].to_numpy(dtype=np.float64)
    volumes = frame[
        [f'SIGNED_VOLUME_{index}' for index in range(20, 0, -1)]
    ].to_numpy(dtype=np.float64)
    paired = np.isfinite(returns) & np.isfinite(volumes)
    count = paired.sum(axis=1)
    returns_zero = np.where(paired, returns, 0.0)
    volumes_zero = np.where(paired, volumes, 0.0)
    r_mean = np.divide(
        returns_zero.sum(axis=1), count,
        out=np.full(len(frame), np.nan), where=count > 0
    )
    v_mean = np.divide(
        volumes_zero.sum(axis=1), count,
        out=np.full(len(frame), np.nan), where=count > 0
    )
    r_centered = np.where(paired, returns - r_mean[:, None], 0.0)
    v_centered = np.where(paired, volumes - v_mean[:, None], 0.0)
    covariance = np.divide(
        (r_centered * v_centered).sum(axis=1), count,
        out=np.full(len(frame), np.nan), where=count >= 5
    )
    r_variance = np.divide(
        (r_centered**2).sum(axis=1), count,
        out=np.full(len(frame), np.nan), where=count >= 5
    )
    v_variance = np.divide(
        (v_centered**2).sum(axis=1), count,
        out=np.full(len(frame), np.nan), where=count >= 5
    )
    denominator = np.sqrt(r_variance * v_variance)
    correlation = np.divide(
        covariance, denominator, out=np.full(len(frame), np.nan),
        where=denominator > EPSILON
    )
    product = returns * volumes
    with np.errstate(all='ignore'):
        mean_product = np.nanmean(product, axis=1)
        mean_abs_product = np.nanmean(np.abs(product), axis=1)
    output = pd.DataFrame(
        {
            'RETURN_VOLUME_CORRELATION': correlation,
            'RETURN_VOLUME_COVARIANCE': covariance,
            'RETURN_VOLUME_MEAN_PRODUCT': mean_product,
            'RETURN_VOLUME_MEAN_ABS_PRODUCT': mean_abs_product,
        }
    )
    output['RETURN_VOLUME_CORRELATION__TS_PERCENTILE'] = _percentile(
        output['RETURN_VOLUME_CORRELATION'], frame['TS']
    )
    output['RETURN_VOLUME_CORRELATION__TS_ROBUST_Z'] = _robust_z(
        output['RETURN_VOLUME_CORRELATION'], frame['TS']
    )
    return output[list(VOLUME_RETURN_INTERACTION_COLUMNS)].astype('float32')


def _turnover_block(
    frame: pd.DataFrame,
    returns: pd.DataFrame,
    cross_returns: pd.DataFrame,
    volume_aggregates: pd.DataFrame,
) -> pd.DataFrame:
    turnover = pd.to_numeric(
        frame['MEDIAN_DAILY_TURNOVER'], errors='raise'
    ).astype('float64')
    subgroup_mean = turnover.groupby(
        _subgroups(frame), observed=True
    ).transform('mean')
    output = pd.DataFrame(
        {
            'TURNOVER_RAW': turnover,
            'TURNOVER_SIGNED_LOG1P': np.sign(turnover)
            * np.log1p(np.abs(turnover)),
            'TURNOVER_MISSING': turnover.isna().astype(float),
            'TURNOVER__TS_PERCENTILE': _percentile(turnover, frame['TS']),
            'TURNOVER__TS_ROBUST_Z': _robust_z(turnover, frame['TS']),
            'TURNOVER__TS_GROUP_DELTA_MEAN': turnover - subgroup_mean,
            'TURNOVER_X_RET_MEAN_W3': turnover * returns['RET_MEAN_W3'],
            'TURNOVER_X_RET_VOLATILITY_W20': turnover
            * returns['RET_VOLATILITY_W20'],
            'TURNOVER_X_RET_1_TS_DELTA_MEAN': turnover
            * cross_returns['RET_1__TS_DELTA_MEAN'],
            'TURNOVER_X_VOLUME_OBSERVED_SHARE': turnover
            * volume_aggregates['VOLUME_OBSERVED_SHARE'],
        }
    )
    return output[list(TURNOVER_COLUMNS)].astype('float32')


def build_feature_blocks(
    frame: pd.DataFrame,
    *,
    partition_name: str,
) -> Mapping[str, pd.DataFrame]:
    '''Build every block without consulting any target column.'''
    canonical = _validate_source(frame, partition_name)
    return_raw, return_dynamics = build_return_blocks(canonical)
    returns = pd.concat([return_raw, return_dynamics], axis=1)
    cross_returns = _cross_sectional_return_block(canonical, returns)
    volume_aggregates, volume_raw, volume_masks = _volume_aggregates(
        canonical
    )
    volume_interactions = _volume_return_interactions(canonical)
    turnover = _turnover_block(
        canonical, returns, cross_returns, volume_aggregates
    )
    blocks: dict[str, pd.DataFrame] = {
        'return_raw': return_raw,
        'return_dynamics': return_dynamics,
        'return_cross_sectional': cross_returns,
        'volume_aggregates': volume_aggregates,
        'volume_raw': volume_raw,
        'volume_missing_masks': volume_masks,
        'volume_return_interactions': volume_interactions,
        'turnover': turnover,
        'group': canonical[['GROUP']].astype('string'),
        'allocation': canonical[['ALLOCATION']].astype('string'),
    }
    for block_name, specification in FEATURE_BLOCKS.items():
        observed = tuple(blocks[block_name].columns)
        if observed != specification.columns:
            raise ValueError(
                f'Block {block_name} column order differs from the registry.'
            )
        if specification.numeric:
            values = blocks[block_name].to_numpy(dtype=np.float64)
            if np.isinf(values).any():
                raise ValueError(f'Block {block_name} contains infinity.')
            if not specification.allows_missing and np.isnan(values).any():
                raise ValueError(
                    f'Block {block_name} contains forbidden missing values.'
                )
    return blocks
