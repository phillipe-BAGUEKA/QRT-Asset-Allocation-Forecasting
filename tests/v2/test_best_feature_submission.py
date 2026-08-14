from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrt_forecasting.v2.feature_families import (
    build_feature_family_pipeline,
    feature_set_by_id,
)
from qrt_forecasting.v2.submission import (
    build_submission_frame,
    fit_full_training_pipeline,
    positive_class_probabilities,
    validate_feature_study_gate,
)


def _valid_study_summary() -> dict:
    reproducibility = {
        'real_second_run_performed': True,
        'canonical_prediction_match': True,
        'run_1_oof_sha256': 'same',
        'run_2_oof_sha256': 'same',
    }
    return {
        'evaluation_scope': 'development_oof_only',
        'assignment_sha256': 'assignment',
        'second_submission_gate_passed': True,
        'lockbox_metrics_computed': False,
        'selected_submission_candidate': 'GB_BEST_NUMERIC_GROUP',
        'experiments': [
            {
                'experiment_id': 'GB_BEST_NUMERIC_GROUP',
                'gate': {'admissible': True},
                'reproducibility': reproducibility,
            }
        ],
    }


def test_feature_study_gate_requires_admission_and_real_reproducibility(
    tmp_path: Path,
) -> None:
    summary = _valid_study_summary()
    (tmp_path / 'study_summary.json').write_text(
        json.dumps(summary), encoding='utf-8'
    )

    loaded, selected = validate_feature_study_gate(
        tmp_path, assignment_sha256='assignment'
    )

    assert loaded == summary
    assert selected == 'GB_BEST_NUMERIC_GROUP'

    summary['experiments'][0]['reproducibility'][
        'real_second_run_performed'
    ] = False
    (tmp_path / 'study_summary.json').write_text(
        json.dumps(summary), encoding='utf-8'
    )
    with pytest.raises(ValueError, match='reproducibility'):
        validate_feature_study_gate(tmp_path, assignment_sha256='assignment')


def test_selected_feature_set_and_full_train_submission_are_exact() -> None:
    best_numeric = feature_set_by_id('GB_RET_TURNOVER')
    selected = feature_set_by_id(
        'GB_BEST_NUMERIC_GROUP', best_numeric=best_numeric
    )
    assert selected.input_features == [
        *[f'RET_{index}' for index in range(1, 21)],
        'MEDIAN_DAILY_TURNOVER',
        'GROUP',
    ]
    rows = []
    for row_id in range(40):
        row = {
            'ROW_ID': row_id,
            'class': row_id % 2,
            'MEDIAN_DAILY_TURNOVER': float(row_id),
            'GROUP': row_id % 4,
        }
        row.update(
            {f'RET_{index}': float(row_id % 2) for index in range(1, 21)}
        )
        rows.append(row)
    data = pd.DataFrame(rows)
    pipeline, fit_time = fit_full_training_pipeline(
        data,
        feature_columns=selected.input_features,
        pipeline_factory=lambda: build_feature_family_pipeline(selected),
        expected_n_rows=40,
    )
    test = data.iloc[:3][['ROW_ID', *selected.input_features]].copy()
    probabilities = positive_class_probabilities(
        pipeline, test[selected.input_features]
    )
    sample = pd.DataFrame({'ROW_ID': test['ROW_ID'], 'prediction': 0})
    submission, prediction_column = build_submission_frame(
        test, sample, (probabilities >= 0.5).astype('int8')
    )

    assert fit_time >= 0.0
    assert prediction_column == 'prediction'
    assert submission.columns.tolist() == ['ROW_ID', 'prediction']
    assert set(submission['prediction']).issubset({0, 1})


def test_runner_validates_gate_before_loading_test_data() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / 'scripts'
        / 'v2'
        / 'build_best_feature_submission.py'
    ).read_text(encoding='utf-8')

    assert source.index('validate_feature_study_gate(') < source.index(
        '_load_test_inputs(selected.input_features)'
    )
