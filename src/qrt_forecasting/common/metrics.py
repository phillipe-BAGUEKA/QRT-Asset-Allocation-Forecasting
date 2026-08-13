'''Reusable binary-classification metrics with strict input validation.'''

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score


def binary_classification_metrics(
    y_true: Any,
    y_proba: Any,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    '''Compute stable binary diagnostics from positive-class probabilities.'''
    truth = np.asarray(y_true)
    probabilities = np.asarray(y_proba, dtype=float)
    if truth.ndim != 1 or probabilities.ndim != 1:
        raise ValueError('y_true and y_proba must be one-dimensional.')
    if len(truth) == 0 or len(truth) != len(probabilities):
        raise ValueError('y_true and y_proba must have equal non-zero lengths.')
    if set(np.unique(truth).tolist()).difference({0, 1}):
        raise ValueError('y_true must contain only binary values 0 and 1.')
    if not np.isfinite(probabilities).all():
        raise ValueError('y_proba must contain only finite values.')
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError('y_proba values must lie in [0, 1].')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('threshold must lie in [0, 1].')

    predictions = (probabilities >= threshold).astype('int8')
    tn, fp, fn, tp = confusion_matrix(
        truth, predictions, labels=[0, 1]
    ).ravel()
    unique_classes = np.unique(truth)
    roc_auc = (
        float(roc_auc_score(truth, probabilities))
        if len(unique_classes) == 2
        else None
    )
    quantiles = np.quantile(
        probabilities, [0.05, 0.25, 0.50, 0.75, 0.95]
    )
    return {
        'n_rows': int(len(truth)),
        'accuracy': float(accuracy_score(truth, predictions)),
        'roc_auc': roc_auc,
        'log_loss': float(log_loss(truth, probabilities, labels=[0, 1])),
        'true_positive_rate': float(np.mean(truth)),
        'predicted_positive_rate': float(np.mean(predictions)),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp),
        'probability_min': float(np.min(probabilities)),
        'probability_max': float(np.max(probabilities)),
        'probability_mean': float(np.mean(probabilities)),
        'probability_std': float(np.std(probabilities)),
        'probability_q05': float(quantiles[0]),
        'probability_q25': float(quantiles[1]),
        'probability_median': float(quantiles[2]),
        'probability_q75': float(quantiles[3]),
        'probability_q95': float(quantiles[4]),
    }
