from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qrt_forecasting.v2.gradient_boosting import (
    build_gradient_boosting_reference_pipeline,
)
from qrt_forecasting.v2.submission import (
    build_submission_frame,
    fit_full_training_pipeline,
    infer_prediction_column,
    persist_validated_submission,
    validate_reference_gate,
)


def test_prediction_column_is_inferred_from_official_shape() -> None:
    sample = pd.DataFrame({'ROW_ID': [10, 11], 'prediction': [0, 0]})

    assert infer_prediction_column(sample) == 'prediction'

    with pytest.raises(ValueError, match='exactly one'):
        infer_prediction_column(pd.DataFrame({'ROW_ID': [10], 'a': [0], 'b': [0]}))


def test_submission_preserves_schema_row_ids_and_order() -> None:
    test = pd.DataFrame({'ROW_ID': [12, 10, 11], 'RET_1': [0.1, 0.2, 0.3]})
    sample = pd.DataFrame({'ROW_ID': [12, 10, 11], 'prediction': [0, 0, 0]})

    submission, prediction_column = build_submission_frame(
        test, sample, np.array([1, 0, 1], dtype='int8')
    )

    assert prediction_column == 'prediction'
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission['ROW_ID'].tolist() == [12, 10, 11]
    assert submission['prediction'].tolist() == [1, 0, 1]


def test_submission_rejects_sample_without_row_id() -> None:
    test = pd.DataFrame({'ROW_ID': [1], 'RET_1': [0.1]})
    sample = pd.DataFrame({'prediction': [0]})

    with pytest.raises(ValueError, match='sample_submission.*ROW_ID'):
        build_submission_frame(test, sample, np.array([0]))


@pytest.mark.parametrize(
    ('test', 'sample', 'predictions', 'message'),
    [
        (
            pd.DataFrame({'ROW_ID': [1, 1]}),
            pd.DataFrame({'ROW_ID': [1, 1], 'prediction': [0, 0]}),
            np.array([0, 1]),
            'duplicate',
        ),
        (
            pd.DataFrame({'ROW_ID': [1, 2]}),
            pd.DataFrame({'ROW_ID': [2, 1], 'prediction': [0, 0]}),
            np.array([0, 1]),
            'order',
        ),
        (
            pd.DataFrame({'ROW_ID': [1, 2]}),
            pd.DataFrame({'ROW_ID': [1, 2], 'prediction': [0, 0]}),
            np.array([0, 2]),
            'only 0 and 1',
        ),
    ],
)
def test_invalid_submission_inputs_are_rejected(test, sample, predictions, message) -> None:
    with pytest.raises(ValueError, match=message):
        build_submission_frame(test, sample, predictions)


def test_gate_requires_matching_successful_artifacts(tmp_path: Path) -> None:
    gate = {'technical_status': 'PASS', 'reproducible': True}
    summary = {
        'gate': gate,
        'experiment_id': 'GB_RET20_REFERENCE',
        'configuration_sha256': 'config',
        'assignment_sha256': 'assignment',
        'evaluation_scope': 'development_oof_only',
        'lockbox_metrics_computed': False,
        'n_development_rows': 421654,
    }
    (tmp_path / 'local_gate.json').write_text(json.dumps(gate), encoding='utf-8')
    (tmp_path / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')

    loaded = validate_reference_gate(
        tmp_path,
        experiment_id='GB_RET20_REFERENCE',
        configuration_sha256='config',
        assignment_sha256='assignment',
    )
    assert loaded == summary

    gate['technical_status'] = 'FAIL'
    (tmp_path / 'local_gate.json').write_text(json.dumps(gate), encoding='utf-8')
    with pytest.raises(ValueError, match='did not pass'):
        validate_reference_gate(
            tmp_path,
            experiment_id='GB_RET20_REFERENCE',
            configuration_sha256='config',
            assignment_sha256='assignment',
        )


def test_full_training_uses_a_fresh_pipeline_and_all_rows() -> None:
    features = [f'RET_{index}' for index in range(1, 21)]
    rows = []
    for row_id in range(20):
        row = {'ROW_ID': row_id, 'class': row_id % 2}
        row.update({feature: float(row_id % 2) for feature in features})
        rows.append(row)
    data = pd.DataFrame(rows)

    pipeline, fit_time = fit_full_training_pipeline(
        data,
        feature_columns=features,
        pipeline_factory=build_gradient_boosting_reference_pipeline,
        expected_n_rows=20,
    )

    assert fit_time >= 0.0
    assert pipeline.n_features_in_ == 20
    assert hasattr(pipeline.named_steps['classifier'], 'estimators_')


def test_persisted_submission_is_re_read_and_hashed(tmp_path: Path) -> None:
    submission = pd.DataFrame({'ROW_ID': [1, 2], 'prediction': pd.Series([0, 1], dtype='int8')})
    path = tmp_path / 'submission.csv'

    digest = persist_validated_submission(submission, path)

    assert len(digest) == 64
    pd.testing.assert_frame_equal(
        pd.read_csv(path),
        pd.DataFrame({'ROW_ID': [1, 2], 'prediction': [0, 1]}),
    )


def test_runner_checks_gate_before_loading_test_data() -> None:
    runner_path = (
        Path(__file__).resolve().parents[2]
        / 'scripts'
        / 'v2'
        / 'build_gb_reference_submission.py'
    )
    source = runner_path.read_text(encoding='utf-8')

    assert source.index('summary = validate_reference_gate(') < source.index(
        'test_features, sample_submission = _load_test_inputs(feature_columns)'
    )
