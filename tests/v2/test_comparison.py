from __future__ import annotations

import pandas as pd
import pytest

from qrt_forecasting.v2.comparison import (
    candidate_gate,
    paired_group_bootstrap_accuracy_gain,
)


def _predictions(candidate_values: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame(
        {
            'ROW_ID': range(8),
            'TS': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            'y_true': [0, 1, 0, 1, 0, 1, 0, 1],
            'y_pred': [0, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    candidate = reference.copy()
    candidate['y_pred'] = candidate_values
    return reference, candidate


def test_group_bootstrap_is_paired_and_deterministic() -> None:
    reference, candidate = _predictions([0, 1, 0, 1, 0, 1, 0, 1])
    reference['y_true'] = reference['y_true'].astype('int64')
    candidate['y_true'] = candidate['y_true'].astype('int8')

    first = paired_group_bootstrap_accuracy_gain(
        reference, candidate, n_iterations=500, seed=42
    )
    second = paired_group_bootstrap_accuracy_gain(
        reference, candidate, n_iterations=500, seed=42
    )

    assert first == second
    assert first['n_groups'] == 4
    assert first['point_accuracy_gain'] == pytest.approx(0.5)
    assert first['ci95_lower'] == pytest.approx(0.5)
    assert first['ci95_upper'] == pytest.approx(0.5)


def test_paired_comparison_rejects_different_row_coverage() -> None:
    reference, candidate = _predictions([0, 1, 0, 1, 0, 1, 0, 1])

    with pytest.raises(ValueError, match='identical ROW_ID'):
        paired_group_bootstrap_accuracy_gain(
            reference, candidate.iloc[:-1], n_iterations=10, seed=1
        )


def test_candidate_gate_requires_gain_robustness_and_log_loss() -> None:
    passed = candidate_gate(
        candidate_accuracy=0.522,
        candidate_log_loss=0.692,
        reference_accuracy=0.5200045,
        reference_log_loss=0.6922,
        reference_accuracy_gate=0.520005,
        minimum_accuracy_gain=0.001,
        fold_accuracy_gains=[0.001, 0.002, 0.001, 0.003, -0.001],
        bootstrap_ci95_lower=-0.001,
        maximum_log_loss_degradation=0.005,
        technical_status='PASS',
    )
    failed = candidate_gate(
        candidate_accuracy=0.5205,
        candidate_log_loss=0.692,
        reference_accuracy=0.5200045,
        reference_log_loss=0.6922,
        reference_accuracy_gate=0.520005,
        minimum_accuracy_gain=0.001,
        fold_accuracy_gains=[0.001] * 5,
        bootstrap_ci95_lower=0.0,
        maximum_log_loss_degradation=0.005,
        technical_status='PASS',
    )

    assert passed['admissible'] is True
    assert failed['admissible'] is False
    assert failed['conditions']['minimum_accuracy_gain_passed'] is False
