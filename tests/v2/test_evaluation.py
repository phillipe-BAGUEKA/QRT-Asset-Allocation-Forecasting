from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from qrt_forecasting.v2.evaluation import (
    canonical_oof_sha256,
    evaluate_grouped_oof,
    evaluate_local_gate,
)
from qrt_forecasting.v2.gradient_boosting import (
    build_gradient_boosting_reference_pipeline,
)


FEATURES = [f'RET_{index}' for index in range(1, 21)]


def _synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_rows = []
    assignment_rows = []
    rng = np.random.default_rng(42)
    row_id = 0
    for group_index in range(15):
        role = 'development' if group_index < 10 else 'lockbox'
        fold_id = group_index % 5 if role == 'development' else pd.NA
        for within_group in range(8):
            target = (group_index + within_group) % 2
            signal = (2 * target - 1) + rng.normal(0.0, 0.1)
            row = {'ROW_ID': row_id, 'TS': f'TS_{group_index:02d}', 'class': target}
            row.update({feature: signal + index * 0.001 for index, feature in enumerate(FEATURES)})
            if role == 'development':
                data_rows.append(row)
            assignment_rows.append(
                {'ROW_ID': row_id, 'TS': row['TS'], 'role': role, 'fold_id': fold_id}
            )
            row_id += 1
    assignment = pd.DataFrame(assignment_rows)
    assignment['fold_id'] = assignment['fold_id'].astype('Int64')
    return pd.DataFrame(data_rows), assignment


def test_grouped_oof_uses_fresh_pipelines_and_predicts_each_row_once() -> None:
    data, assignment = _synthetic_data()
    pipelines = []

    def factory():
        pipeline = build_gradient_boosting_reference_pipeline()
        pipelines.append(pipeline)
        return pipeline

    result = evaluate_grouped_oof(
        data,
        assignment,
        feature_columns=FEATURES,
        pipeline_factory=factory,
    )

    assert len(pipelines) == 5
    assert len({id(pipeline) for pipeline in pipelines}) == 5
    assert all(hasattr(p.named_steps['classifier'], 'estimators_') for p in pipelines)
    assert len(result.oof_predictions) == len(data)
    assert result.oof_predictions['ROW_ID'].is_unique
    assert sorted(result.fold_metrics['fold_id'].tolist()) == [0, 1, 2, 3, 4]
    assert result.fold_metrics['n_validation'].sum() == len(data)
    assert result.fold_metrics['fit_time_seconds'].ge(0).all()
    assert result.fold_metrics['prediction_time_seconds'].ge(0).all()
    assert result.global_metrics['accuracy'] == pytest.approx(
        result.global_metrics['weighted_fold_accuracy']
    )


def test_two_runs_are_exactly_reproducible_and_pass_the_gate() -> None:
    data, assignment = _synthetic_data()
    arguments = dict(
        feature_columns=FEATURES,
        pipeline_factory=build_gradient_boosting_reference_pipeline,
    )
    first = evaluate_grouped_oof(data, assignment, **arguments)
    second = evaluate_grouped_oof(data, assignment, **arguments)

    gate = evaluate_local_gate(first, second)

    assert gate['technical_status'] == 'PASS'
    assert gate['reproducible'] is True
    assert first.oof_sha256 == second.oof_sha256
    np.testing.assert_array_equal(
        first.oof_predictions['y_proba'], second.oof_predictions['y_proba']
    )


def test_each_imputer_is_fitted_on_its_training_fold_only() -> None:
    data, assignment = _synthetic_data()
    pipelines = []

    class RecordingImputer(SimpleImputer):
        def fit(self, X, y=None):
            self.fit_indices_ = set(X.index.tolist())
            return super().fit(X, y)

    def factory():
        pipeline = Pipeline(
            [
                ('imputer', RecordingImputer(strategy='constant', fill_value=0.0)),
                (
                    'classifier',
                    GradientBoostingClassifier(
                        n_estimators=2, max_depth=1, random_state=42
                    ),
                ),
            ]
        )
        pipelines.append(pipeline)
        return pipeline

    evaluate_grouped_oof(
        data,
        assignment,
        feature_columns=FEATURES,
        pipeline_factory=factory,
    )

    fold_by_row = assignment.loc[
        assignment['role'] == 'development', ['ROW_ID', 'fold_id']
    ].set_index('ROW_ID')['fold_id']
    for fold_id, pipeline in enumerate(pipelines):
        expected_indices = set(
            data.index[data['ROW_ID'].map(fold_by_row).ne(fold_id)].tolist()
        )
        assert pipeline.named_steps['imputer'].fit_indices_ == expected_indices


def test_lockbox_row_in_oof_input_is_rejected() -> None:
    data, assignment = _synthetic_data()
    lockbox_row = data.iloc[[0]].copy()
    lockbox_id = assignment.loc[assignment['role'] == 'lockbox', 'ROW_ID'].iloc[0]
    lockbox_ts = assignment.loc[assignment['ROW_ID'] == lockbox_id, 'TS'].iloc[0]
    lockbox_row['ROW_ID'] = lockbox_id
    lockbox_row['TS'] = lockbox_ts

    with pytest.raises(ValueError, match='Lockbox'):
        evaluate_grouped_oof(
            pd.concat([data, lockbox_row], ignore_index=True),
            assignment,
            feature_columns=FEATURES,
            pipeline_factory=build_gradient_boosting_reference_pipeline,
        )


def test_group_overlap_is_rejected() -> None:
    data, assignment = _synthetic_data()
    development_rows = assignment['role'] == 'development'
    first_id = assignment.loc[development_rows & (assignment['fold_id'] == 0), 'ROW_ID'].iloc[0]
    second_id = assignment.loc[development_rows & (assignment['fold_id'] == 1), 'ROW_ID'].iloc[0]
    assignment.loc[assignment['ROW_ID'] == second_id, 'TS'] = assignment.loc[
        assignment['ROW_ID'] == first_id, 'TS'
    ].iloc[0]
    data.loc[data['ROW_ID'] == second_id, 'TS'] = data.loc[
        data['ROW_ID'] == first_id, 'TS'
    ].iloc[0]

    with pytest.raises(ValueError, match='leaks'):
        evaluate_grouped_oof(
            data,
            assignment,
            feature_columns=FEATURES,
            pipeline_factory=build_gradient_boosting_reference_pipeline,
        )


def test_canonical_oof_hash_is_independent_of_row_order() -> None:
    data, assignment = _synthetic_data()
    result = evaluate_grouped_oof(
        data,
        assignment,
        feature_columns=FEATURES,
        pipeline_factory=build_gradient_boosting_reference_pipeline,
    )

    assert result.oof_sha256 == canonical_oof_sha256(
        result.oof_predictions.sample(frac=1.0, random_state=7)
    )
