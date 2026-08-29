from __future__ import annotations

import numpy as np
import pandas as pd

from qrt_forecasting.v2.boosting import (
    CatBoostNativePreprocessor,
    LightGBMAdapter,
    SelectionFit,
    SparseOneHotPreprocessor,
    StableNativeCategoryPreprocessor,
    XGBoostAdapter,
    boosting_adapters,
    make_nested_fold_indices,
    run_engine_on_outer_fold,
)


def identity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'ROW_ID': range(10),
            'TS': [index // 2 for index in range(10)],
            'fold_id': [index // 2 for index in range(10)],
        }
    )


def test_nested_grouped_partitions_follow_frozen_rule() -> None:
    data = identity()
    partitions = make_nested_fold_indices(data, outer_fold_id=3)
    assert partitions.internal_validation_fold_id == 4
    assert set(data.iloc[partitions.outer_validation]['fold_id']) == {3}
    assert set(data.iloc[partitions.internal_validation]['fold_id']) == {4}
    assert set(data.iloc[partitions.internal_train]['fold_id']) == {0, 1, 2}
    assert set(data.iloc[partitions.outer_train]['fold_id']) == {0, 1, 2, 4}


def test_preprocessors_fit_only_authorized_rows_and_accept_unknown_category() -> None:
    train = pd.DataFrame(
        {'numeric': [1.0, np.nan], 'category': ['A', 'B']},
        index=[10, 11],
    )
    validation = pd.DataFrame(
        {'numeric': [3.0], 'category': ['UNKNOWN']},
        index=[99],
    )
    sparse = SparseOneHotPreprocessor(
        ['numeric'], ['category']
    ).fit(train)
    transformed = sparse.transform(validation)
    assert sparse.fit_index_ == (10, 11)
    assert transformed.shape == (1, 3)
    assert transformed.format == 'csr'

    native = StableNativeCategoryPreprocessor(
        ['numeric'], ['category']
    ).fit(train)
    native_validation = native.transform(validation)
    assert native.fit_index_ == (10, 11)
    assert str(native_validation['category'].iloc[0]) == '__UNKNOWN__'
    assert native_validation['category'].dtype.name == 'category'

    catboost = CatBoostNativePreprocessor(
        ['numeric'], ['category']
    ).fit(train)
    assert catboost.fit_index_ == (10, 11)
    assert catboost.transform(validation)['category'].iloc[0] == 'UNKNOWN'


class RecordingAdapter:
    name = 'recording'

    def select(
        self, train, y_train, validation, y_validation,
        numeric_columns, categorical_columns,
    ):
        self.internal_train_index = tuple(train.index)
        self.internal_validation_index = tuple(validation.index)
        return SelectionFit(object(), object(), 3, 0.01)

    def refit_predict(
        self, selection, train, y_train, validation,
        numeric_columns, categorical_columns,
    ):
        self.outer_train_index = tuple(train.index)
        self.outer_validation_index = tuple(validation.index)
        return (
            np.linspace(0.2, 0.8, len(validation)),
            object(),
            object(),
            0.02,
            0.003,
        )


def test_runner_uses_authorized_scopes_and_fresh_refit_objects() -> None:
    data = identity()
    matrix = pd.DataFrame(
        {'feature': np.arange(10, dtype=np.float32)}
    )
    target = np.array([0, 1] * 5)
    adapter = RecordingAdapter()
    result = run_engine_on_outer_fold(
        adapter,
        matrix,
        target,
        data,
        outer_fold_id=0,
        numeric_columns=['feature'],
        categorical_columns=[],
        peak_rss_bytes=123,
    )
    assert set(adapter.internal_train_index) == set(range(4, 10))
    assert set(adapter.internal_validation_index) == {2, 3}
    assert set(adapter.outer_train_index) == set(range(2, 10))
    assert set(adapter.outer_validation_index) == {0, 1}
    assert result.estimator_objects_are_distinct is True
    assert result.preprocessing_objects_are_distinct is True
    assert result.best_iteration == 3


def test_exact_engine_versions_and_cpu_factories() -> None:
    xgb, lgb, cat = boosting_adapters(max_estimators=5, patience=2)
    assert [xgb.version, lgb.version, cat.version] == [
        '3.3.0', '4.6.0', '1.2.10'
    ]
    xgb_first = xgb._estimator(n_estimators=5, early_stopping=True)
    xgb_second = xgb._estimator(n_estimators=5, early_stopping=True)
    assert xgb_first.get_params()['device'] == 'cpu'
    assert xgb_first.get_params()['n_jobs'] == 1
    assert xgb_first.get_params()['callbacks'][0] is not (
        xgb_second.get_params()['callbacks'][0]
    )
    lgb_model = lgb._estimator(5)
    assert lgb_model.get_params()['deterministic'] is True
    assert (
        lgb_model.get_params()['force_col_wise']
        != lgb_model.get_params()['force_row_wise']
    )
    cat_model = cat._estimator(5)
    assert cat_model.get_params()['task_type'] == 'CPU'
    assert cat_model.get_params()['thread_count'] == 1


def test_lightgbm_layout_flags_remain_mutually_exclusive() -> None:
    adapter = LightGBMAdapter(max_estimators=5, patience=2)
    adapter.force_col_wise = False
    model = adapter._estimator(5)
    assert model.get_params()['force_col_wise'] is False
    assert model.get_params()['force_row_wise'] is True
