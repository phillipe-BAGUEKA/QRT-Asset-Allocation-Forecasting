'''Leakage-safe out-of-fold evaluation for frozen grouped V2 folds.'''

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from qrt_forecasting.common.metrics import binary_classification_metrics
from qrt_forecasting.v2.folds import DEVELOPMENT_ROLE, LOCKBOX_ROLE


OOF_COLUMNS = ['ROW_ID', 'TS', 'fold_id', 'y_true', 'y_proba', 'y_pred']


@dataclass(frozen=True)
class OOFRunResult:
    '''Complete diagnostics from one development-only OOF execution.'''

    oof_predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    group_metrics: pd.DataFrame
    probability_histogram: pd.DataFrame
    global_metrics: dict[str, Any]
    total_fit_time_seconds: float
    total_prediction_time_seconds: float
    oof_sha256: str


def canonical_oof_sha256(oof_predictions: pd.DataFrame) -> str:
    '''Hash canonical OOF predictions at full round-trip float precision.'''
    missing = sorted(set(OOF_COLUMNS).difference(oof_predictions.columns))
    if missing:
        raise ValueError(f'Cannot hash OOF predictions; missing columns: {missing}.')
    canonical = oof_predictions[OOF_COLUMNS].sort_values(
        'ROW_ID', kind='mergesort'
    )
    payload = canonical.to_csv(
        index=False, lineterminator='\n', float_format='%.17g'
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _validate_inputs(
    development_data: pd.DataFrame,
    assignment: pd.DataFrame,
    feature_columns: Sequence[str],
    n_splits: int,
) -> pd.DataFrame:
    required_data = {'ROW_ID', 'TS', 'class', *feature_columns}
    missing_data = sorted(required_data.difference(development_data.columns))
    if missing_data:
        raise ValueError(f'Development data is missing columns: {missing_data}.')
    required_assignment = {'ROW_ID', 'TS', 'role', 'fold_id'}
    missing_assignment = sorted(required_assignment.difference(assignment.columns))
    if missing_assignment:
        raise ValueError(f'Assignment is missing columns: {missing_assignment}.')
    if development_data['ROW_ID'].isna().any():
        raise ValueError('Development data contains missing ROW_ID values.')
    if development_data['ROW_ID'].duplicated().any():
        raise ValueError('Development data contains duplicate ROW_ID values.')

    development_assignment = assignment[assignment['role'] == DEVELOPMENT_ROLE]
    lockbox_ids = set(
        assignment.loc[assignment['role'] == LOCKBOX_ROLE, 'ROW_ID'].tolist()
    )
    data_ids = set(development_data['ROW_ID'].tolist())
    expected_ids = set(development_assignment['ROW_ID'].tolist())
    if data_ids.intersection(lockbox_ids):
        raise ValueError('Lockbox rows are forbidden in OOF evaluation.')
    if data_ids != expected_ids or len(development_data) != len(development_assignment):
        raise ValueError('Development data does not exactly cover frozen development IDs.')

    merged = development_data.merge(
        development_assignment[['ROW_ID', 'TS', 'fold_id']],
        on='ROW_ID',
        how='left',
        suffixes=('_data', '_assignment'),
        validate='one_to_one',
    )
    if not merged['TS_data'].equals(merged['TS_assignment']):
        raise ValueError('At least one TS differs from the frozen assignment.')
    merged = merged.drop(columns=['TS_assignment']).rename(columns={'TS_data': 'TS'})
    if merged['fold_id'].isna().any():
        raise ValueError('At least one development row has no fold ID.')
    merged['fold_id'] = merged['fold_id'].astype('int64')
    observed_folds = sorted(merged['fold_id'].unique().tolist())
    if observed_folds != list(range(n_splits)):
        raise ValueError(f'Expected folds 0..{n_splits - 1}; found {observed_folds}.')
    if set(merged['class'].unique().tolist()) != {0, 1}:
        raise ValueError('Development class must contain exactly 0 and 1.')
    return merged


def _positive_class_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(model.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if len(positive_positions) != 1:
        raise ValueError('Fitted estimator must expose exactly one positive class 1.')
    return probabilities[:, int(positive_positions[0])]


def _validate_oof_predictions(
    oof: pd.DataFrame,
    expected_ids: set[Any],
) -> None:
    if len(oof) != len(expected_ids):
        raise ValueError('OOF coverage is incomplete.')
    if oof['ROW_ID'].isna().any() or oof['ROW_ID'].duplicated().any():
        raise ValueError('OOF ROW_ID values must be present and unique.')
    if set(oof['ROW_ID'].tolist()) != expected_ids:
        raise ValueError('OOF ROW_ID coverage differs from development.')
    probabilities = oof['y_proba'].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError('OOF probabilities must be finite.')
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError('OOF probabilities must lie in [0, 1].')
    if set(oof['y_pred'].unique().tolist()).difference({0, 1}):
        raise ValueError('OOF predictions must contain only 0 and 1.')


def evaluate_grouped_oof(
    development_data: pd.DataFrame,
    assignment: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    pipeline_factory: Callable[[], Pipeline],
    threshold: float = 0.5,
    n_splits: int = 5,
    compute_group_metrics: bool = True,
) -> OOFRunResult:
    '''Fit one fresh pipeline per fold and predict each development row once.'''
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must lie in [0, 1].')
    merged = _validate_inputs(
        development_data, assignment, feature_columns, n_splits
    )
    fold_frames: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    total_fit_time = 0.0
    total_prediction_time = 0.0

    for fold_id in range(n_splits):
        train = merged[merged['fold_id'] != fold_id]
        validation = merged[merged['fold_id'] == fold_id]
        overlap = set(train['TS']).intersection(validation['TS'])
        if overlap:
            raise ValueError(f'Fold {fold_id} leaks {len(overlap)} TS groups.')

        pipeline = pipeline_factory()
        if not isinstance(pipeline, Pipeline):
            raise TypeError('pipeline_factory must return an sklearn Pipeline.')
        fit_started = time.perf_counter()
        pipeline.fit(train[list(feature_columns)], train['class'])
        fit_time = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = _positive_class_probabilities(
            pipeline, validation[list(feature_columns)]
        )
        prediction_time = time.perf_counter() - prediction_started
        predictions = (probabilities >= threshold).astype('int8')
        metrics = binary_classification_metrics(
            validation['class'].to_numpy(), probabilities, threshold=threshold
        )
        metrics.update(
            {
                'fold_id': fold_id,
                'n_train': int(len(train)),
                'n_validation': int(len(validation)),
                'n_train_groups': int(train['TS'].nunique()),
                'n_validation_groups': int(validation['TS'].nunique()),
                'fit_time_seconds': fit_time,
                'prediction_time_seconds': prediction_time,
            }
        )
        fold_metrics.append(metrics)
        fold_frames.append(
            pd.DataFrame(
                {
                    'ROW_ID': validation['ROW_ID'].to_numpy(),
                    'TS': validation['TS'].to_numpy(),
                    'fold_id': fold_id,
                    'y_true': validation['class'].to_numpy(dtype='int8'),
                    'y_proba': probabilities,
                    'y_pred': predictions,
                }
            )
        )
        total_fit_time += fit_time
        total_prediction_time += prediction_time

    oof = pd.concat(fold_frames, ignore_index=True).sort_values(
        'ROW_ID', kind='mergesort'
    ).reset_index(drop=True)
    _validate_oof_predictions(oof, set(merged['ROW_ID'].tolist()))
    global_metrics = binary_classification_metrics(
        oof['y_true'].to_numpy(), oof['y_proba'].to_numpy(), threshold=threshold
    )
    weighted_accuracy = float(
        np.average(
            [row['accuracy'] for row in fold_metrics],
            weights=[row['n_validation'] for row in fold_metrics],
        )
    )
    if not np.isclose(global_metrics['accuracy'], weighted_accuracy, atol=1e-15):
        raise ValueError('Global accuracy differs from weighted fold accuracy.')
    global_metrics['weighted_fold_accuracy'] = weighted_accuracy

    group_rows: list[dict[str, Any]] = []
    if compute_group_metrics:
        for group, group_frame in oof.groupby('TS', sort=True, observed=True):
            group_row = binary_classification_metrics(
                group_frame['y_true'].to_numpy(),
                group_frame['y_proba'].to_numpy(),
                threshold=threshold,
            )
            group_row.update(
                {'TS': group, 'fold_id': int(group_frame['fold_id'].iloc[0])}
            )
            group_rows.append(group_row)
    group_metrics = pd.DataFrame(group_rows)

    counts, edges = np.histogram(oof['y_proba'], bins=np.linspace(0.0, 1.0, 21))
    histogram = pd.DataFrame(
        {
            'bin_left': edges[:-1],
            'bin_right': edges[1:],
            'count': counts.astype('int64'),
        }
    )
    return OOFRunResult(
        oof_predictions=oof,
        fold_metrics=pd.DataFrame(fold_metrics).sort_values('fold_id'),
        group_metrics=group_metrics,
        probability_histogram=histogram,
        global_metrics=global_metrics,
        total_fit_time_seconds=total_fit_time,
        total_prediction_time_seconds=total_prediction_time,
        oof_sha256=canonical_oof_sha256(oof),
    )


def evaluate_local_gate(
    first: OOFRunResult,
    second: OOFRunResult,
) -> dict[str, Any]:
    '''Enforce technical reproducibility and return non-blocking warnings.'''
    if first.oof_sha256 != second.oof_sha256:
        raise ValueError('Deterministic OOF hashes diverged.')
    first_probabilities = first.oof_predictions['y_proba'].to_numpy()
    second_probabilities = second.oof_predictions['y_proba'].to_numpy()
    if not np.array_equal(first_probabilities, second_probabilities):
        raise ValueError('Deterministic OOF probabilities diverged.')
    if first.oof_predictions['y_pred'].nunique() < 2:
        raise ValueError('OOF class predictions are constant.')

    metrics = first.global_metrics
    majority_accuracy = max(
        metrics['true_positive_rate'], 1.0 - metrics['true_positive_rate']
    )
    warnings: list[str] = []
    if metrics['accuracy'] <= majority_accuracy + 0.001:
        warnings.append('accuracy_proche_ou_inferieure_classe_majoritaire')
    if metrics['probability_std'] < 0.01:
        warnings.append('probabilites_tres_concentrees')
    predicted_rate = metrics['predicted_positive_rate']
    if predicted_rate < 0.10 or predicted_rate > 0.90:
        warnings.append('classes_predites_desequilibrees')
    fold_accuracy_range = float(
        first.fold_metrics['accuracy'].max() - first.fold_metrics['accuracy'].min()
    )
    if fold_accuracy_range > 0.02:
        warnings.append('variation_accuracy_inter_folds_elevee')
    return {
        'technical_status': 'PASS',
        'reproducible': True,
        'oof_sha256': first.oof_sha256,
        'majority_class_accuracy': majority_accuracy,
        'accuracy_gain_over_majority': metrics['accuracy'] - majority_accuracy,
        'scientific_warnings': warnings,
    }
