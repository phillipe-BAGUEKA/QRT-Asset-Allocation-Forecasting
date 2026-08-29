'''Reproducible development-only Parquet cache for V2 features.'''

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow

from qrt_forecasting.v2.feature_engineering import build_feature_blocks
from qrt_forecasting.v2.feature_registry import (
    FEATURE_BLOCKS,
    FEATURE_SETS,
    feature_columns,
    feature_columns_sha256,
)
from qrt_forecasting.v2.folds import (
    ASSIGNMENT_COLUMNS,
    DEVELOPMENT_ROLE,
    LOCKBOX_ROLE,
    canonical_dataframe_sha256,
)
from qrt_forecasting.v2.gradient_boosting import file_sha256


CACHE_SCHEMA_VERSION = 1
IDENTITY_COLUMNS = ('ROW_ID', 'TS', 'fold_id')


def canonical_block_sha256(frame: pd.DataFrame) -> str:
    '''Hash values, dtypes and column order independently of Parquet metadata.'''
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                'columns': list(frame.columns),
                'dtypes': [str(dtype) for dtype in frame.dtypes],
            },
            separators=(',', ':'),
        ).encode('utf-8')
    )
    for column in frame.columns:
        hashes = pd.util.hash_pandas_object(
            frame[column], index=False, categorize=True
        ).to_numpy(dtype=np.uint64)
        digest.update(hashes.tobytes())
    return digest.hexdigest()


def _duplicate_and_constant_columns(
    blocks: dict[str, pd.DataFrame],
) -> tuple[list[list[str]], list[str]]:
    fingerprints: dict[str, list[tuple[str, pd.Series]]] = {}
    constants: list[str] = []
    for block in blocks.values():
        for column in block.columns:
            values = block[column]
            if values.nunique(dropna=False) <= 1:
                constants.append(column)
            fingerprint = hashlib.sha256(
                pd.util.hash_pandas_object(
                    values, index=False, categorize=True
                ).to_numpy(dtype=np.uint64).tobytes()
            ).hexdigest()
            fingerprints.setdefault(fingerprint, []).append((column, values))
    duplicates: list[list[str]] = []
    for candidates in fingerprints.values():
        if len(candidates) < 2:
            continue
        used: set[str] = set()
        for index, (name, values) in enumerate(candidates):
            if name in used:
                continue
            group = [name]
            for other_name, other_values in candidates[index + 1 :]:
                if values.equals(other_values):
                    group.append(other_name)
                    used.add(other_name)
            if len(group) > 1:
                duplicates.append(group)
    return sorted(duplicates), sorted(constants)


def _select_development(
    raw_features: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    expected_rows: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(set(ASSIGNMENT_COLUMNS).difference(assignment.columns))
    if missing:
        raise ValueError(f'Assignment is missing columns: {missing}.')
    if assignment['ROW_ID'].isna().any() or assignment['ROW_ID'].duplicated().any():
        raise ValueError('Assignment ROW_ID values must be present and unique.')
    roles = set(assignment['role'].unique().tolist())
    if roles != {DEVELOPMENT_ROLE, LOCKBOX_ROLE}:
        raise ValueError(f'Unexpected assignment roles: {roles}.')
    development_assignment = assignment[
        assignment['role'] == DEVELOPMENT_ROLE
    ][list(ASSIGNMENT_COLUMNS)].copy()
    if expected_rows is not None and len(development_assignment) != expected_rows:
        raise ValueError(
            f'Expected {expected_rows} development rows; '
            f'found {len(development_assignment)}.'
        )
    if development_assignment['fold_id'].isna().any():
        raise ValueError('Development assignment contains missing fold IDs.')
    if sorted(development_assignment['fold_id'].astype(int).unique()) != list(
        range(5)
    ):
        raise ValueError('The frozen development folds must be exactly 0..4.')
    if raw_features['ROW_ID'].isna().any() or raw_features['ROW_ID'].duplicated().any():
        raise ValueError('Raw ROW_ID values must be present and unique.')
    selected = development_assignment.merge(
        raw_features,
        on='ROW_ID',
        how='left',
        suffixes=('_assignment', ''),
        validate='one_to_one',
    )
    if selected.isna().all(axis=1).any():
        raise ValueError('At least one development row is missing from raw data.')
    if 'TS_assignment' in selected:
        if not selected['TS_assignment'].equals(selected['TS']):
            raise ValueError('Raw TS values differ from the frozen assignment.')
        selected = selected.drop(columns='TS_assignment')
    lockbox_ids = set(
        assignment.loc[assignment['role'] == LOCKBOX_ROLE, 'ROW_ID'].tolist()
    )
    if set(selected['ROW_ID']).intersection(lockbox_ids):
        raise ValueError('Lockbox rows are forbidden in the feature cache.')
    selected['fold_id'] = selected['fold_id'].astype('int64')
    selected = selected.sort_values('ROW_ID', kind='mergesort').reset_index(
        drop=True
    )
    development_assignment = development_assignment.sort_values(
        'ROW_ID', kind='mergesort'
    ).reset_index(drop=True)
    return selected, development_assignment


def build_development_feature_cache(
    raw_features: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    output_directory: Path,
    source_path: Path | None = None,
    expected_rows: int | None = 421_654,
) -> dict[str, Any]:
    '''Build a Zstandard Parquet cache containing development rows only.'''
    output_directory = Path(output_directory).resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f'Cache directory is not empty: {output_directory}.'
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    selected, development_assignment = _select_development(
        raw_features, assignment, expected_rows=expected_rows
    )
    blocks = dict(
        build_feature_blocks(selected, partition_name=DEVELOPMENT_ROLE)
    )
    identity = selected[list(IDENTITY_COLUMNS)].copy()
    duplicate_columns, constant_columns = _duplicate_and_constant_columns(blocks)

    identity_path = output_directory / 'identity.parquet'
    identity.to_parquet(
        identity_path, index=False, compression='zstd', engine='pyarrow'
    )
    block_manifest: dict[str, Any] = {}
    for block_name, block in blocks.items():
        cached = pd.concat([selected[['ROW_ID']], block], axis=1)
        block_path = output_directory / f'{block_name}.parquet'
        cached.to_parquet(
            block_path, index=False, compression='zstd', engine='pyarrow'
        )
        block_manifest[block_name] = {
            'file': block_path.name,
            'n_rows': int(len(cached)),
            'n_features': int(block.shape[1]),
            'columns': list(block.columns),
            'columns_sha256': feature_columns_sha256(tuple(block.columns)),
            'content_sha256': canonical_block_sha256(cached),
            'file_sha256': file_sha256(block_path),
            'size_bytes': block_path.stat().st_size,
        }

    source_hash = (
        file_sha256(source_path)
        if source_path is not None
        else canonical_block_sha256(
            selected.drop(columns=['role', 'fold_id'], errors='ignore')
        )
    )
    manifest: dict[str, Any] = {
        'schema_version': CACHE_SCHEMA_VERSION,
        'scope': 'development_only',
        'n_rows': int(len(selected)),
        'source_sha256': source_hash,
        'row_ids_sha256': canonical_dataframe_sha256(
            selected, columns=['ROW_ID', 'TS']
        ),
        'folds_sha256': canonical_dataframe_sha256(
            development_assignment,
            columns=['ROW_ID', 'TS', 'role', 'fold_id'],
        ),
        'identity': {
            'file': identity_path.name,
            'content_sha256': canonical_block_sha256(identity),
            'file_sha256': file_sha256(identity_path),
            'size_bytes': identity_path.stat().st_size,
        },
        'blocks': block_manifest,
        'feature_sets': {
            name: {
                'blocks': list(specification.blocks),
                'n_features': len(feature_columns(name)),
                'columns_sha256': feature_columns_sha256(feature_columns(name)),
            }
            for name, specification in FEATURE_SETS.items()
        },
        'duplicate_columns': duplicate_columns,
        'constant_columns': constant_columns,
        'format': {'type': 'parquet', 'compression': 'zstd'},
        'versions': {
            'python': platform.python_version(),
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'pyarrow': pyarrow.__version__,
        },
        'lockbox_rows_cached': 0,
        'test_data_accessed': False,
    }
    manifest_path = output_directory / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
        newline='\n',
    )
    return manifest


def load_cached_feature_set(
    cache_directory: Path,
    feature_set_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    '''Load and verify one declared feature matrix from cached blocks.'''
    cache_directory = Path(cache_directory).resolve()
    manifest_path = cache_directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('scope') != 'development_only':
        raise ValueError('Only a development-only cache may be loaded.')
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f'Unknown feature set: {feature_set_name}.')
    identity_path = cache_directory / manifest['identity']['file']
    if file_sha256(identity_path) != manifest['identity']['file_sha256']:
        raise ValueError('Cached identity file hash changed.')
    identity = pd.read_parquet(identity_path)
    matrices: list[pd.DataFrame] = []
    for block_name in FEATURE_SETS[feature_set_name].blocks:
        details = manifest['blocks'][block_name]
        path = cache_directory / details['file']
        if file_sha256(path) != details['file_sha256']:
            raise ValueError(f'Cached block {block_name} file hash changed.')
        cached = pd.read_parquet(path)
        if not cached['ROW_ID'].equals(identity['ROW_ID']):
            raise ValueError(f'Cached block {block_name} ROW_ID order changed.')
        matrices.append(cached.drop(columns='ROW_ID'))
    matrix = pd.concat(matrices, axis=1)
    if tuple(matrix.columns) != feature_columns(feature_set_name):
        raise ValueError('Cached feature order differs from the registry.')
    return identity, matrix, manifest
