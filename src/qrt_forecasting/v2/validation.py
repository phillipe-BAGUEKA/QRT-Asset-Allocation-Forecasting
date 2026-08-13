'''Deterministic grouped-split construction and structural auditing.'''

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


ROW_ID_COLUMN = 'ROW_ID'
GROUP_COLUMN = 'TS'
TARGET_COLUMN = 'class'
FOLD_COLUMN = 'fold_id'
SUPPORTED_SPLITTERS = ('GroupKFold', 'StratifiedGroupKFold')


def validate_grouped_source(data: pd.DataFrame) -> pd.DataFrame:
    '''Validate and canonically order labelled rows for grouped splitting.'''
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame.')
    if data.empty:
        raise ValueError('data must not be empty.')

    required = {ROW_ID_COLUMN, GROUP_COLUMN, TARGET_COLUMN}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f'Missing grouped-validation columns: {missing}.')
    if data[ROW_ID_COLUMN].isna().any():
        raise ValueError('ROW_ID contains missing values.')
    if data[ROW_ID_COLUMN].duplicated().any():
        raise ValueError('ROW_ID contains duplicate values.')
    if data[GROUP_COLUMN].isna().any():
        raise ValueError('TS contains missing group identifiers.')
    if data[TARGET_COLUMN].isna().any():
        raise ValueError('class contains missing values.')

    classes = set(data[TARGET_COLUMN].unique().tolist())
    if classes != {0, 1}:
        raise ValueError(f'class must contain exactly 0 and 1; found {classes}.')

    # Sorting is only a canonicalization device. It gives TS no temporal meaning.
    return data.sort_values(
        [GROUP_COLUMN, ROW_ID_COLUMN],
        kind='mergesort',
    ).reset_index(drop=True)


def make_grouped_splitter(
    splitter_name: str,
    *,
    n_splits: int,
    seed: int,
) -> GroupKFold | StratifiedGroupKFold:
    '''Create one supported shuffled grouped splitter.'''
    if splitter_name not in SUPPORTED_SPLITTERS:
        raise ValueError(
            f'Unsupported splitter {splitter_name}; expected {SUPPORTED_SPLITTERS}.'
        )
    if isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError('n_splits must be an integer greater than one.')

    splitter_type = (
        GroupKFold
        if splitter_name == 'GroupKFold'
        else StratifiedGroupKFold
    )
    return splitter_type(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )


def create_group_fold_assignment(
    data: pd.DataFrame,
    *,
    splitter_name: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    '''Assign folds through TS membership, independently of source row order.'''
    canonical = validate_grouped_source(data)
    if canonical[GROUP_COLUMN].nunique() < n_splits:
        raise ValueError('The number of TS groups must be at least n_splits.')

    splitter = make_grouped_splitter(
        splitter_name,
        n_splits=n_splits,
        seed=seed,
    )
    group_to_fold: dict[Any, int] = {}
    dummy_features = np.zeros((len(canonical), 1), dtype=np.int8)

    for fold_id, (_, validation_positions) in enumerate(
        splitter.split(
            dummy_features,
            canonical[TARGET_COLUMN],
            groups=canonical[GROUP_COLUMN],
        )
    ):
        # sklearn exposes row positions, but only group IDs are retained. Final
        # membership is always assigned with the TS-to-fold mapping below.
        validation_groups = canonical.iloc[validation_positions][
            GROUP_COLUMN
        ].unique()
        for group in validation_groups:
            if group in group_to_fold:
                raise ValueError(f'TS group {group} was assigned more than once.')
            group_to_fold[group] = fold_id

    assignment = canonical[[ROW_ID_COLUMN, GROUP_COLUMN]].copy()
    assignment[FOLD_COLUMN] = assignment[GROUP_COLUMN].map(group_to_fold)
    if assignment[FOLD_COLUMN].isna().any():
        raise ValueError('At least one row was not assigned to a fold.')
    assignment[FOLD_COLUMN] = assignment[FOLD_COLUMN].astype('int64')
    return assignment.sort_values(ROW_ID_COLUMN, kind='mergesort').reset_index(
        drop=True
    )


def audit_fold_assignment(
    data: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    n_splits: int,
) -> dict[str, Any]:
    '''Check grouped invariants and return predeclared balance diagnostics.'''
    canonical = validate_grouped_source(data)
    required_assignment = {ROW_ID_COLUMN, GROUP_COLUMN, FOLD_COLUMN}
    missing = sorted(required_assignment.difference(assignment.columns))
    if missing:
        raise ValueError(f'Missing assignment columns: {missing}.')
    if assignment[ROW_ID_COLUMN].isna().any():
        raise ValueError('Assignment ROW_ID contains missing values.')
    if assignment[ROW_ID_COLUMN].duplicated().any():
        raise ValueError('Assignment ROW_ID contains duplicate values.')

    merged = canonical.merge(
        assignment[[ROW_ID_COLUMN, GROUP_COLUMN, FOLD_COLUMN]],
        on=ROW_ID_COLUMN,
        how='left',
        suffixes=('_source', '_assignment'),
        validate='one_to_one',
    )
    violations: list[str] = []
    if len(assignment) != len(canonical):
        violations.append('row_count_mismatch')
    if merged[FOLD_COLUMN].isna().any():
        violations.append('incomplete_row_coverage')
    if not merged[f'{GROUP_COLUMN}_source'].equals(
        merged[f'{GROUP_COLUMN}_assignment']
    ):
        violations.append('group_identifier_mismatch')

    observed_folds = sorted(merged[FOLD_COLUMN].dropna().astype(int).unique())
    if observed_folds != list(range(n_splits)):
        violations.append('incorrect_fold_identifiers')

    group_fold_counts = merged.groupby(
        f'{GROUP_COLUMN}_source', observed=True
    )[FOLD_COLUMN].nunique()
    if not group_fold_counts.eq(1).all():
        violations.append('group_overlap')

    global_prevalence = float(canonical[TARGET_COLUMN].mean())
    expected_rows = len(canonical) / n_splits
    expected_groups = canonical[GROUP_COLUMN].nunique() / n_splits
    fold_statistics: list[dict[str, Any]] = []

    for fold_id in range(n_splits):
        validation = merged[merged[FOLD_COLUMN] == fold_id]
        training = merged[merged[FOLD_COLUMN] != fold_id]
        validation_classes = set(validation[TARGET_COLUMN].unique().tolist())
        training_classes = set(training[TARGET_COLUMN].unique().tolist())
        if validation_classes != {0, 1}:
            violations.append(f'fold_{fold_id}_validation_missing_class')
        if training_classes != {0, 1}:
            violations.append(f'fold_{fold_id}_training_missing_class')

        n_rows = int(len(validation))
        n_groups = int(validation[f'{GROUP_COLUMN}_source'].nunique())
        positive_rate = float(validation[TARGET_COLUMN].mean())
        fold_statistics.append(
            {
                'fold_id': fold_id,
                'n_rows': n_rows,
                'n_groups': n_groups,
                'n_negative': int((validation[TARGET_COLUMN] == 0).sum()),
                'n_positive': int((validation[TARGET_COLUMN] == 1).sum()),
                'positive_rate': positive_rate,
                'relative_row_deviation': abs(n_rows - expected_rows)
                / expected_rows,
                'relative_group_deviation': abs(n_groups - expected_groups)
                / expected_groups,
                'absolute_prevalence_deviation': abs(
                    positive_rate - global_prevalence
                ),
            }
        )

    unique_violations = sorted(set(violations))
    max_row = max(item['relative_row_deviation'] for item in fold_statistics)
    max_group = max(
        item['relative_group_deviation'] for item in fold_statistics
    )
    max_prevalence = max(
        item['absolute_prevalence_deviation'] for item in fold_statistics
    )
    return {
        'valid': not unique_violations,
        'violations': unique_violations,
        'n_rows': int(len(canonical)),
        'n_groups': int(canonical[GROUP_COLUMN].nunique()),
        'global_positive_rate': global_prevalence,
        'max_relative_row_deviation': max_row,
        'max_relative_group_deviation': max_group,
        'max_absolute_prevalence_deviation': max_prevalence,
        'worst_balance_deviation': max(max_row, max_group, max_prevalence),
        'total_balance_deviation': max_row + max_group + max_prevalence,
        'folds': fold_statistics,
    }


def audit_splitter(
    data: pd.DataFrame,
    *,
    splitter_name: str,
    n_splits: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    '''Build and structurally audit one splitter candidate.'''
    assignment = create_group_fold_assignment(
        data,
        splitter_name=splitter_name,
        n_splits=n_splits,
        seed=seed,
    )
    audit = audit_fold_assignment(data, assignment, n_splits=n_splits)
    audit['splitter'] = splitter_name
    audit['n_splits'] = n_splits
    audit['seed'] = seed
    return assignment, audit


def select_splitter(audits: list[dict[str, Any]]) -> dict[str, Any]:
    '''Select a valid splitter with the predeclared equal-scale criteria.'''
    if not audits:
        raise ValueError('At least one splitter audit is required.')
    valid_audits = [audit for audit in audits if audit.get('valid') is True]
    if not valid_audits:
        raise ValueError('No splitter candidate satisfies the hard invariants.')

    return min(
        valid_audits,
        key=lambda audit: (
            audit['worst_balance_deviation'],
            audit['total_balance_deviation'],
            audit['max_relative_row_deviation'],
            audit['max_relative_group_deviation'],
            audit['max_absolute_prevalence_deviation'],
            audit['splitter'],
        ),
    )
