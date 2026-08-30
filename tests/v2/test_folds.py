from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from qrt_forecasting.v2.folds import (
    ASSIGNMENT_COLUMNS,
    build_manifest,
    canonical_dataframe_sha256,
    create_development_lockbox_assignment,
    load_assignment_artifacts,
    persist_assignment_artifacts,
    validate_final_assignment,
)
from qrt_forecasting.v2.validation import audit_splitter


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


def _final_assignment() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    data = _grouped_data()
    assignment, details = create_development_lockbox_assignment(
        data,
        splitter_name='StratifiedGroupKFold',
        seed=42,
        lockbox_n_splits=5,
        lockbox_target_fraction=0.20,
        development_n_splits=5,
    )
    return data, assignment, details


def test_lockbox_and_development_folds_keep_every_group_intact() -> None:
    data, assignment, details = _final_assignment()

    assert assignment.columns.tolist() == ASSIGNMENT_COLUMNS
    assert assignment['ROW_ID'].nunique() == len(data)
    assert assignment.groupby('TS')['role'].nunique().eq(1).all()
    assert assignment.loc[assignment['role'] == 'lockbox', 'fold_id'].isna().all()
    assert assignment.loc[
        assignment['role'] == 'development', 'fold_id'
    ].notna().all()
    assert sorted(
        assignment.loc[
            assignment['role'] == 'development', 'fold_id'
        ].astype(int).unique()
    ) == [0, 1, 2, 3, 4]
    assert details['lockbox']['row_fraction'] == pytest.approx(0.20)
    assert details['development']['row_fraction'] == pytest.approx(0.80)
    assert details['development_fold_audit']['valid'] is True
    assert validate_final_assignment(
        data, assignment, development_n_splits=5
    )['valid'] is True


def test_final_assignment_is_independent_of_source_row_order() -> None:
    data = _grouped_data()
    first, _ = create_development_lockbox_assignment(
        data,
        splitter_name='StratifiedGroupKFold',
        seed=42,
        lockbox_n_splits=5,
        lockbox_target_fraction=0.20,
        development_n_splits=5,
    )
    second, _ = create_development_lockbox_assignment(
        data.sample(frac=1.0, random_state=7),
        splitter_name='StratifiedGroupKFold',
        seed=42,
        lockbox_n_splits=5,
        lockbox_target_fraction=0.20,
        development_n_splits=5,
    )

    pd.testing.assert_frame_equal(first, second)


def test_canonical_hash_is_stable_under_row_reordering() -> None:
    _, assignment, _ = _final_assignment()
    original_hash = canonical_dataframe_sha256(
        assignment, columns=ASSIGNMENT_COLUMNS
    )
    reordered_hash = canonical_dataframe_sha256(
        assignment.sample(frac=1.0, random_state=11),
        columns=ASSIGNMENT_COLUMNS,
    )

    assert original_hash == reordered_hash


def test_persisted_assignment_is_hash_verified(tmp_path: Path) -> None:
    data, assignment, details = _final_assignment()
    _, splitter_audit = audit_splitter(
        data,
        splitter_name='StratifiedGroupKFold',
        n_splits=5,
        seed=42,
    )
    configuration = {
        'seed': 42,
        'audit_n_splits': 5,
        'lockbox_target_fraction': 0.20,
        'development_n_splits': 5,
    }
    manifest = build_manifest(
        data=data,
        assignment=assignment,
        configuration=configuration,
        splitter_audits=[splitter_audit],
        assignment_details=details,
    )
    assignment_path = tmp_path / 'assignment.csv'
    manifest_path = tmp_path / 'manifest.json'

    persist_assignment_artifacts(
        assignment=assignment,
        manifest=manifest,
        assignment_path=assignment_path,
        manifest_path=manifest_path,
    )
    loaded_assignment, loaded_manifest = load_assignment_artifacts(
        assignment_path, manifest_path
    )

    pd.testing.assert_frame_equal(assignment, loaded_assignment)
    assert loaded_manifest['assignment_sha256'] == manifest['assignment_sha256']
    assert loaded_manifest['development_identifiers_sha256']
    assert loaded_manifest['lockbox_identifiers_sha256']
    assert loaded_manifest['data_fingerprint_sha256']
    assert loaded_manifest['versions']['scikit_learn']

    tampered = loaded_assignment.copy()
    development_index = tampered.index[tampered['role'] == 'development'][0]
    tampered.loc[development_index, 'fold_id'] = (
        int(tampered.loc[development_index, 'fold_id']) + 1
    ) % 5
    tampered.to_csv(assignment_path, index=False)
    with pytest.raises(ValueError, match='hash'):
        load_assignment_artifacts(assignment_path, manifest_path)
