'''Lockbox, development-fold and artifact management for V2.'''

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from qrt_forecasting.v2.validation import (
    FOLD_COLUMN,
    GROUP_COLUMN,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
    audit_fold_assignment,
    create_group_fold_assignment,
    validate_grouped_source,
)


ROLE_COLUMN = 'role'
DEVELOPMENT_ROLE = 'development'
LOCKBOX_ROLE = 'lockbox'
ASSIGNMENT_COLUMNS = [
    ROW_ID_COLUMN,
    GROUP_COLUMN,
    ROLE_COLUMN,
    FOLD_COLUMN,
]


def canonical_dataframe_sha256(
    frame: pd.DataFrame,
    *,
    columns: list[str],
) -> str:
    '''Hash selected columns after stable ROW_ID ordering and normalization.'''
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f'Cannot hash missing columns: {missing}.')
    if ROW_ID_COLUMN not in columns:
        raise ValueError('Canonical hashes must include ROW_ID.')

    canonical = frame[columns].copy().sort_values(
        ROW_ID_COLUMN, kind='mergesort'
    )
    for column in columns:
        if column == FOLD_COLUMN:
            canonical[column] = canonical[column].map(
                lambda value: '' if pd.isna(value) else str(int(value))
            )
        else:
            canonical[column] = canonical[column].astype(str)
    payload = canonical.to_csv(
        index=False,
        lineterminator='\n',
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _partition_statistics(
    data: pd.DataFrame,
    row_ids: set[Any],
) -> dict[str, Any]:
    partition = data[data[ROW_ID_COLUMN].isin(row_ids)]
    return {
        'n_rows': int(len(partition)),
        'n_groups': int(partition[GROUP_COLUMN].nunique()),
        'n_negative': int((partition[TARGET_COLUMN] == 0).sum()),
        'n_positive': int((partition[TARGET_COLUMN] == 1).sum()),
        'positive_rate': float(partition[TARGET_COLUMN].mean()),
        'row_fraction': float(len(partition) / len(data)),
        'group_fraction': float(
            partition[GROUP_COLUMN].nunique() / data[GROUP_COLUMN].nunique()
        ),
    }


def create_development_lockbox_assignment(
    data: pd.DataFrame,
    *,
    splitter_name: str,
    seed: int,
    lockbox_n_splits: int,
    lockbox_target_fraction: float,
    development_n_splits: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    '''Freeze one grouped lockbox, then create folds on development only.'''
    canonical = validate_grouped_source(data)
    if not 0.0 < lockbox_target_fraction < 0.5:
        raise ValueError('lockbox_target_fraction must lie in (0, 0.5).')

    lockbox_candidates = create_group_fold_assignment(
        canonical,
        splitter_name=splitter_name,
        n_splits=lockbox_n_splits,
        seed=seed,
    )
    candidate_frame = canonical.merge(
        lockbox_candidates,
        on=[ROW_ID_COLUMN, GROUP_COLUMN],
        how='inner',
        validate='one_to_one',
    )
    global_prevalence = float(canonical[TARGET_COLUMN].mean())
    lockbox_options: list[dict[str, Any]] = []

    for candidate_fold in range(lockbox_n_splits):
        candidate = candidate_frame[
            candidate_frame[FOLD_COLUMN] == candidate_fold
        ]
        option = {
            'candidate_fold_id': candidate_fold,
            'n_rows': int(len(candidate)),
            'n_groups': int(candidate[GROUP_COLUMN].nunique()),
            'positive_rate': float(candidate[TARGET_COLUMN].mean()),
            'row_fraction': float(len(candidate) / len(canonical)),
            'group_fraction': float(
                candidate[GROUP_COLUMN].nunique()
                / canonical[GROUP_COLUMN].nunique()
            ),
        }
        option['absolute_row_fraction_deviation'] = abs(
            option['row_fraction'] - lockbox_target_fraction
        )
        option['absolute_group_fraction_deviation'] = abs(
            option['group_fraction'] - lockbox_target_fraction
        )
        option['absolute_prevalence_deviation'] = abs(
            option['positive_rate'] - global_prevalence
        )
        lockbox_options.append(option)

    selected_option = min(
        lockbox_options,
        key=lambda item: (
            item['absolute_row_fraction_deviation'],
            item['absolute_group_fraction_deviation'],
            item['absolute_prevalence_deviation'],
            item['candidate_fold_id'],
        ),
    )
    lockbox_groups = set(
        candidate_frame.loc[
            candidate_frame[FOLD_COLUMN]
            == selected_option['candidate_fold_id'],
            GROUP_COLUMN,
        ].unique()
    )
    lockbox_mask = canonical[GROUP_COLUMN].isin(lockbox_groups)
    development_data = canonical.loc[~lockbox_mask].copy()
    lockbox_data = canonical.loc[lockbox_mask].copy()

    development_folds = create_group_fold_assignment(
        development_data,
        splitter_name=splitter_name,
        n_splits=development_n_splits,
        seed=seed,
    )
    development_fold_map = development_folds.set_index(ROW_ID_COLUMN)[
        FOLD_COLUMN
    ]

    assignment = canonical[[ROW_ID_COLUMN, GROUP_COLUMN]].copy()
    assignment[ROLE_COLUMN] = np.where(
        assignment[GROUP_COLUMN].isin(lockbox_groups),
        LOCKBOX_ROLE,
        DEVELOPMENT_ROLE,
    )
    assignment[FOLD_COLUMN] = assignment[ROW_ID_COLUMN].map(
        development_fold_map
    ).astype('Int64')
    assignment = assignment[ASSIGNMENT_COLUMNS].sort_values(
        ROW_ID_COLUMN, kind='mergesort'
    ).reset_index(drop=True)

    development_audit = validate_final_assignment(
        canonical,
        assignment,
        development_n_splits=development_n_splits,
    )
    details = {
        'selected_splitter': splitter_name,
        'seed': seed,
        'lockbox_target_fraction': lockbox_target_fraction,
        'lockbox_selected_candidate_fold': selected_option[
            'candidate_fold_id'
        ],
        'lockbox_candidates': lockbox_options,
        'development': _partition_statistics(
            canonical,
            set(development_data[ROW_ID_COLUMN].tolist()),
        ),
        'lockbox': _partition_statistics(
            canonical,
            set(lockbox_data[ROW_ID_COLUMN].tolist()),
        ),
        'development_fold_audit': development_audit,
    }
    return assignment, details


def validate_final_assignment(
    data: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    development_n_splits: int,
) -> dict[str, Any]:
    '''Validate lockbox isolation and all definitive development folds.'''
    canonical = validate_grouped_source(data)
    missing = sorted(set(ASSIGNMENT_COLUMNS).difference(assignment.columns))
    if missing:
        raise ValueError(f'Missing final assignment columns: {missing}.')
    if len(assignment) != len(canonical):
        raise ValueError('Final assignment row count differs from source data.')
    if assignment[ROW_ID_COLUMN].isna().any():
        raise ValueError('Final assignment contains missing ROW_ID values.')
    if assignment[ROW_ID_COLUMN].duplicated().any():
        raise ValueError('Final assignment contains duplicate ROW_ID values.')

    merged = canonical.merge(
        assignment[ASSIGNMENT_COLUMNS],
        on=ROW_ID_COLUMN,
        how='left',
        suffixes=('_source', '_assignment'),
        validate='one_to_one',
    )
    if merged[ROLE_COLUMN].isna().any():
        raise ValueError('Final assignment does not cover every source row.')
    if not merged[f'{GROUP_COLUMN}_source'].equals(
        merged[f'{GROUP_COLUMN}_assignment']
    ):
        raise ValueError('Final assignment changed at least one TS value.')
    roles = set(merged[ROLE_COLUMN].unique().tolist())
    if roles != {DEVELOPMENT_ROLE, LOCKBOX_ROLE}:
        raise ValueError(f'Final assignment roles are invalid: {roles}.')

    group_role_counts = merged.groupby(
        f'{GROUP_COLUMN}_source', observed=True
    )[ROLE_COLUMN].nunique()
    if not group_role_counts.eq(1).all():
        raise ValueError('At least one TS group crosses development and lockbox.')

    lockbox = merged[merged[ROLE_COLUMN] == LOCKBOX_ROLE]
    development = merged[merged[ROLE_COLUMN] == DEVELOPMENT_ROLE]
    if lockbox.empty or development.empty:
        raise ValueError('Development and lockbox must both be non-empty.')
    if lockbox[FOLD_COLUMN].notna().any():
        raise ValueError('Lockbox rows must not receive development fold IDs.')
    if development[FOLD_COLUMN].isna().any():
        raise ValueError('Every development row must receive a fold ID.')

    development_source = development[
        [ROW_ID_COLUMN, f'{GROUP_COLUMN}_source', TARGET_COLUMN]
    ].rename(columns={f'{GROUP_COLUMN}_source': GROUP_COLUMN})
    development_assignment = development[
        [ROW_ID_COLUMN, f'{GROUP_COLUMN}_source', FOLD_COLUMN]
    ].rename(columns={f'{GROUP_COLUMN}_source': GROUP_COLUMN})
    development_assignment[FOLD_COLUMN] = development_assignment[
        FOLD_COLUMN
    ].astype('int64')
    return audit_fold_assignment(
        development_source,
        development_assignment,
        n_splits=development_n_splits,
    )


def build_manifest(
    *,
    data: pd.DataFrame,
    assignment: pd.DataFrame,
    configuration: dict[str, Any],
    splitter_audits: list[dict[str, Any]],
    assignment_details: dict[str, Any],
) -> dict[str, Any]:
    '''Build the lightweight tracked manifest for the frozen assignment.'''
    return {
        'schema_version': 1,
        'methodology': 'grouped_development_lockbox',
        'data_fingerprint_sha256': canonical_dataframe_sha256(
            data,
            columns=[ROW_ID_COLUMN, GROUP_COLUMN, TARGET_COLUMN],
        ),
        'assignment_sha256': canonical_dataframe_sha256(
            assignment,
            columns=ASSIGNMENT_COLUMNS,
        ),
        'development_identifiers_sha256': canonical_dataframe_sha256(
            assignment[assignment[ROLE_COLUMN] == DEVELOPMENT_ROLE],
            columns=[ROW_ID_COLUMN, GROUP_COLUMN, ROLE_COLUMN],
        ),
        'lockbox_identifiers_sha256': canonical_dataframe_sha256(
            assignment[assignment[ROLE_COLUMN] == LOCKBOX_ROLE],
            columns=[ROW_ID_COLUMN, GROUP_COLUMN, ROLE_COLUMN],
        ),
        'assignment_columns': ASSIGNMENT_COLUMNS,
        'configuration': configuration,
        'splitter_selection': {
            'criteria_defined_before_audit': True,
            'hard_invariants': [
                'no_group_overlap',
                'complete_row_coverage',
                'both_classes_in_train_and_validation',
            ],
            'balance_components': [
                'max_relative_row_deviation',
                'max_relative_group_deviation',
                'max_absolute_prevalence_deviation',
            ],
            'candidate_audits': splitter_audits,
            'selected_splitter': assignment_details['selected_splitter'],
        },
        'partition': assignment_details,
        'versions': {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit_learn': sklearn.__version__,
        },
    }


def persist_assignment_artifacts(
    *,
    assignment: pd.DataFrame,
    manifest: dict[str, Any],
    assignment_path: Path,
    manifest_path: Path,
) -> None:
    '''Write assignment CSV and tracked JSON manifest atomically.'''
    assignment_path = assignment_path.resolve()
    manifest_path = manifest_path.resolve()
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        newline='',
        suffix='.csv',
        dir=assignment_path.parent,
        delete=False,
    ) as temporary_assignment:
        assignment.to_csv(temporary_assignment, index=False, lineterminator='\n')
        temporary_assignment_path = Path(temporary_assignment.name)
    os.replace(temporary_assignment_path, assignment_path)

    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        newline='\n',
        suffix='.json',
        dir=manifest_path.parent,
        delete=False,
    ) as temporary_manifest:
        json.dump(manifest, temporary_manifest, indent=2, sort_keys=True)
        temporary_manifest.write('\n')
        temporary_manifest_path = Path(temporary_manifest.name)
    os.replace(temporary_manifest_path, manifest_path)


def load_assignment_artifacts(
    assignment_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    '''Load artifacts and reject an assignment whose frozen hash changed.'''
    assignment = pd.read_csv(assignment_path, dtype={ROLE_COLUMN: str})
    assignment[FOLD_COLUMN] = assignment[FOLD_COLUMN].astype('Int64')
    with manifest_path.open(encoding='utf-8') as manifest_file:
        manifest = json.load(manifest_file)
    observed_hash = canonical_dataframe_sha256(
        assignment,
        columns=ASSIGNMENT_COLUMNS,
    )
    expected_hash = manifest.get('assignment_sha256')
    if observed_hash != expected_hash:
        raise ValueError(
            'Assignment hash does not match the frozen validation manifest.'
        )
    return assignment, manifest
