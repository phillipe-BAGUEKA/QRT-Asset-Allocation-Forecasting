from __future__ import annotations

import numpy as np
import pandas as pd

from qrt_forecasting.v2.feature_audit import (
    audit_categorical_overlap,
    audit_feature_columns,
)


def test_feature_audit_reports_numeric_statistics_and_presence() -> None:
    development = pd.DataFrame(
        {'ROW_ID': [1, 2], 'RET_1': [1.0, np.nan], 'GROUP': [1, 2], 'ALLOCATION': ['A', 'B']}
    )
    test = pd.DataFrame(
        {'ROW_ID': [3], 'RET_1': [2.0], 'GROUP': [2], 'ALLOCATION': ['B']}
    )

    audit = audit_feature_columns(development, test).set_index('name')

    assert audit.loc['RET_1', 'family'] == 'RET'
    assert audit.loc['RET_1', 'missing_rate'] == 0.5
    assert audit.loc['RET_1', 'minimum'] == 1.0
    assert bool(audit.loc['RET_1', 'in_test']) is True


def test_category_overlap_and_memory_estimate_are_reproducible() -> None:
    development = pd.DataFrame(
        {'GROUP': [1, 2, 2], 'ALLOCATION': ['A', 'B', 'C']}
    )
    test = pd.DataFrame({'GROUP': [2, 3], 'ALLOCATION': ['B', 'D']})

    first = audit_categorical_overlap(development, test)
    second = audit_categorical_overlap(development, test)

    assert first == second
    assert first[0]['common_count'] == 1
    assert first[0]['development_only_count'] == 1
    assert first[0]['test_only_count'] == 1
    assert first[0]['estimated_sparse_csr_megabytes_per_fold'] > 0
