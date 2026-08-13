from __future__ import annotations

import pandas as pd
import pytest

from qrt_forecasting.v2.validation import (
    audit_splitter,
    create_group_fold_assignment,
    select_splitter,
)


def _grouped_data(n_groups: int = 30, rows_per_group: int = 4) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 0
    for group_index in range(n_groups):
        for within_group in range(rows_per_group):
            rows.append(
                {
                    'ROW_ID': row_id,
                    'TS': f'GROUP_{group_index:03d}',
                    'class': (group_index + within_group) % 2,
                }
            )
            row_id += 1
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    'splitter_name', ['GroupKFold', 'StratifiedGroupKFold']
)
def test_group_assignment_is_stable_and_keeps_groups_intact(
    splitter_name: str,
) -> None:
    data = _grouped_data()
    shuffled = data.sample(frac=1.0, random_state=91).reset_index(drop=True)

    original = create_group_fold_assignment(
        data, splitter_name=splitter_name, n_splits=5, seed=42
    )
    reordered = create_group_fold_assignment(
        shuffled, splitter_name=splitter_name, n_splits=5, seed=42
    )

    pd.testing.assert_frame_equal(original, reordered)
    assert original.groupby('TS')['fold_id'].nunique().eq(1).all()
    assert original['ROW_ID'].nunique() == len(data)


@pytest.mark.parametrize(
    'splitter_name', ['GroupKFold', 'StratifiedGroupKFold']
)
def test_splitter_audit_enforces_structural_invariants(
    splitter_name: str,
) -> None:
    _, audit = audit_splitter(
        _grouped_data(),
        splitter_name=splitter_name,
        n_splits=5,
        seed=42,
    )

    assert audit['valid'] is True
    assert audit['violations'] == []
    assert len(audit['folds']) == 5
    assert sum(fold['n_rows'] for fold in audit['folds']) == 120
    assert sum(fold['n_groups'] for fold in audit['folds']) == 30
    assert all(fold['n_negative'] > 0 for fold in audit['folds'])
    assert all(fold['n_positive'] > 0 for fold in audit['folds'])


def test_splitter_selection_uses_predeclared_balance_scores() -> None:
    weaker = {
        'splitter': 'GroupKFold',
        'valid': True,
        'worst_balance_deviation': 0.20,
        'total_balance_deviation': 0.25,
        'max_relative_row_deviation': 0.20,
        'max_relative_group_deviation': 0.03,
        'max_absolute_prevalence_deviation': 0.02,
    }
    stronger = {
        'splitter': 'StratifiedGroupKFold',
        'valid': True,
        'worst_balance_deviation': 0.10,
        'total_balance_deviation': 0.18,
        'max_relative_row_deviation': 0.10,
        'max_relative_group_deviation': 0.07,
        'max_absolute_prevalence_deviation': 0.01,
    }

    assert select_splitter([weaker, stronger]) is stronger


def test_splitter_selection_rejects_candidates_with_failed_invariants() -> None:
    with pytest.raises(ValueError, match='hard invariants'):
        select_splitter(
            [
                {
                    'splitter': 'GroupKFold',
                    'valid': False,
                    'violations': ['group_overlap'],
                }
            ]
        )
