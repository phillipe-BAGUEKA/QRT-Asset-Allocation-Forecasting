'''Paired grouped comparison and deterministic candidate admission gate.'''

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _paired_correctness_by_group(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    required = ['ROW_ID', 'TS', 'y_true', 'y_pred']
    for frame, name in ((reference, 'reference'), (candidate, 'candidate')):
        missing = sorted(set(required).difference(frame.columns))
        if missing:
            raise ValueError(f'{name} is missing columns: {missing}.')
        if frame['ROW_ID'].duplicated().any():
            raise ValueError(f'{name} contains duplicate ROW_ID values.')
    paired = reference[required].merge(
        candidate[required],
        on='ROW_ID',
        suffixes=('_reference', '_candidate'),
        validate='one_to_one',
    )
    if len(paired) != len(reference) or len(paired) != len(candidate):
        raise ValueError('Paired comparison requires identical ROW_ID coverage.')
    if not np.array_equal(
        paired['TS_reference'].to_numpy(), paired['TS_candidate'].to_numpy()
    ):
        raise ValueError('Paired comparison TS values differ.')
    if not np.array_equal(
        paired['y_true_reference'].to_numpy(dtype=int),
        paired['y_true_candidate'].to_numpy(dtype=int),
    ):
        raise ValueError('Paired comparison targets differ.')
    paired['difference'] = (
        paired['y_pred_candidate'].eq(paired['y_true_candidate']).astype(int)
        - paired['y_pred_reference'].eq(paired['y_true_reference']).astype(int)
    )
    return paired.groupby('TS_reference', sort=True, observed=True).agg(
        difference_sum=('difference', 'sum'),
        n_rows=('ROW_ID', 'size'),
    )


def paired_group_bootstrap_accuracy_gain(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    n_iterations: int,
    seed: int,
) -> dict[str, float | int]:
    '''Bootstrap accuracy differences by resampling complete TS groups.'''
    if n_iterations < 1:
        raise ValueError('n_iterations must be positive.')
    grouped = _paired_correctness_by_group(reference, candidate)
    differences = grouped['difference_sum'].to_numpy(dtype=float)
    counts = grouped['n_rows'].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    gains = np.empty(n_iterations, dtype=float)
    for index in range(n_iterations):
        sampled = rng.integers(0, len(grouped), size=len(grouped))
        gains[index] = differences[sampled].sum() / counts[sampled].sum()
    lower, upper = np.quantile(gains, [0.025, 0.975])
    return {
        'seed': seed,
        'n_iterations': n_iterations,
        'n_groups': int(len(grouped)),
        'point_accuracy_gain': float(differences.sum() / counts.sum()),
        'ci95_lower': float(lower),
        'ci95_upper': float(upper),
    }


def candidate_gate(
    *,
    candidate_accuracy: float,
    candidate_log_loss: float,
    reference_accuracy: float,
    reference_log_loss: float,
    reference_accuracy_gate: float,
    minimum_accuracy_gain: float,
    fold_accuracy_gains: list[float],
    bootstrap_ci95_lower: float,
    maximum_log_loss_degradation: float,
    technical_status: str,
) -> dict[str, Any]:
    '''Apply the predeclared technical and scientific second-CSV gate.'''
    conditions = {
        'technical_gate_passed': technical_status == 'PASS',
        'minimum_accuracy_gain_passed': candidate_accuracy
        >= reference_accuracy_gate + minimum_accuracy_gain,
        'robustness_passed': sum(gain >= 0.0 for gain in fold_accuracy_gains) >= 4
        or bootstrap_ci95_lower >= 0.0,
        'log_loss_passed': candidate_log_loss - reference_log_loss
        <= maximum_log_loss_degradation,
    }
    return {
        'admissible': all(conditions.values()),
        'conditions': conditions,
        'accuracy_gain_vs_reference_artifact': candidate_accuracy
        - reference_accuracy,
        'non_negative_fold_count': sum(
            gain >= 0.0 for gain in fold_accuracy_gains
        ),
        'log_loss_degradation': candidate_log_loss - reference_log_loss,
    }
