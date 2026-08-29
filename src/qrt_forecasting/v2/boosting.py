'''CPU-only boosting adapters with grouped internal early stopping.'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import catboost
import lightgbm
import numpy as np
import pandas as pd
import scipy.sparse
import xgboost
from sklearn.preprocessing import OneHotEncoder

from qrt_forecasting.common.metrics import binary_classification_metrics


RANDOM_STATE = 42
CLASSIFICATION_THRESHOLD = 0.5
N_SPLITS = 5
RAM_LIMIT_BYTES = 12 * 1024**3


@dataclass(frozen=True)
class NestedFoldIndices:
    outer_fold_id: int
    internal_validation_fold_id: int
    internal_train: np.ndarray
    internal_validation: np.ndarray
    outer_train: np.ndarray
    outer_validation: np.ndarray


@dataclass
class SelectionFit:
    estimator: Any
    preprocessor: Any
    best_iteration: int
    duration_seconds: float


@dataclass
class EnginePilotResult:
    engine: str
    outer_fold_id: int
    internal_validation_fold_id: int
    best_iteration: int
    early_stopping_fit_seconds: float
    outer_refit_seconds: float
    prediction_seconds: float
    peak_rss_bytes: int
    metrics: dict[str, Any]
    probabilities: np.ndarray
    estimator_objects_are_distinct: bool
    preprocessing_objects_are_distinct: bool


def make_nested_fold_indices(
    identity: pd.DataFrame,
    *,
    outer_fold_id: int,
    n_splits: int = N_SPLITS,
) -> NestedFoldIndices:
    '''Freeze outer and internal partitions using whole grouped folds.'''
    required = {'ROW_ID', 'TS', 'fold_id'}
    missing = sorted(required.difference(identity.columns))
    if missing:
        raise ValueError(f'Identity is missing columns: {missing}.')
    observed = sorted(identity['fold_id'].astype(int).unique().tolist())
    if observed != list(range(n_splits)):
        raise ValueError(f'Expected frozen folds 0..{n_splits - 1}.')
    if outer_fold_id not in observed:
        raise ValueError(f'Unknown outer fold: {outer_fold_id}.')
    if identity.groupby('TS', observed=True)['fold_id'].nunique().gt(1).any():
        raise ValueError('At least one TS crosses frozen folds.')
    internal_validation_fold_id = (outer_fold_id + 1) % n_splits
    fold_ids = identity['fold_id'].to_numpy(dtype=int)
    outer_validation = np.flatnonzero(fold_ids == outer_fold_id)
    outer_train = np.flatnonzero(fold_ids != outer_fold_id)
    internal_validation = np.flatnonzero(
        fold_ids == internal_validation_fold_id
    )
    internal_train = np.flatnonzero(
        (fold_ids != outer_fold_id)
        & (fold_ids != internal_validation_fold_id)
    )
    sets = [
        set(identity.iloc[positions]['TS'].tolist())
        for positions in (
            outer_validation, internal_validation, internal_train
        )
    ]
    if sets[0].intersection(sets[1]) or sets[0].intersection(sets[2]):
        raise ValueError('Outer validation TS leaked into training.')
    if sets[1].intersection(sets[2]):
        raise ValueError('Internal validation TS leaked into internal training.')
    return NestedFoldIndices(
        outer_fold_id=outer_fold_id,
        internal_validation_fold_id=internal_validation_fold_id,
        internal_train=internal_train,
        internal_validation=internal_validation,
        outer_train=outer_train,
        outer_validation=outer_validation,
    )


class SparseOneHotPreprocessor:
    '''Sparse numeric-plus-OHE transform fitted only on authorized rows.'''

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)
        self.encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=True,
            dtype=np.float32,
        )

    def fit(self, frame: pd.DataFrame) -> 'SparseOneHotPreprocessor':
        self.fit_index_ = tuple(frame.index.tolist())
        if self.categorical_columns:
            self.encoder.fit(frame[self.categorical_columns].astype('string'))
        return self

    def transform(self, frame: pd.DataFrame) -> scipy.sparse.csr_matrix:
        numeric = scipy.sparse.csr_matrix(
            frame[self.numeric_columns].to_numpy(dtype=np.float32)
        )
        if not self.categorical_columns:
            return numeric
        categorical = self.encoder.transform(
            frame[self.categorical_columns].astype('string')
        )
        return scipy.sparse.hstack(
            [numeric, categorical], format='csr', dtype=np.float32
        )


class StableNativeCategoryPreprocessor:
    '''Stable train-fitted pandas categories for native LightGBM handling.'''

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)

    def fit(self, frame: pd.DataFrame) -> 'StableNativeCategoryPreprocessor':
        self.fit_index_ = tuple(frame.index.tolist())
        self.categories_: dict[str, list[str]] = {}
        for column in self.categorical_columns:
            values = frame[column].astype('string').fillna('__MISSING__')
            categories = sorted(values.unique().tolist())
            self.categories_[column] = [
                *categories, '__UNKNOWN__'
            ]
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame[self.numeric_columns].astype('float32').copy()
        for column in self.categorical_columns:
            values = frame[column].astype('string').fillna('__MISSING__')
            categories = self.categories_[column]
            values = values.where(values.isin(categories), '__UNKNOWN__')
            output[column] = pd.Categorical(values, categories=categories)
        return output


class CatBoostNativePreprocessor:
    '''Minimal native-category conversion with auditable fit scope.'''

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> None:
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns = list(categorical_columns)

    def fit(self, frame: pd.DataFrame) -> 'CatBoostNativePreprocessor':
        self.fit_index_ = tuple(frame.index.tolist())
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame[self.numeric_columns].astype('float32').copy()
        for column in self.categorical_columns:
            output[column] = (
                frame[column].astype('string').fillna('__MISSING__')
            )
        return output


class XGBoostAdapter:
    name = 'xgboost'
    version = xgboost.__version__

    def __init__(self, *, max_estimators: int = 400, patience: int = 20):
        self.max_estimators = max_estimators
        self.patience = patience

    def _estimator(self, *, n_estimators: int, early_stopping: bool) -> Any:
        callbacks = None
        if early_stopping:
            # XGBoost callbacks are stateful and must never be reused.
            callbacks = [
                xgboost.callback.EarlyStopping(
                    rounds=self.patience,
                    data_name='validation_0',
                    metric_name='logloss',
                    save_best=False,
                )
            ]
        return xgboost.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            device='cpu',
            n_jobs=1,
            random_state=RANDOM_STATE,
            callbacks=callbacks,
        )

    def select(
        self,
        train: pd.DataFrame,
        y_train: np.ndarray,
        validation: pd.DataFrame,
        y_validation: np.ndarray,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> SelectionFit:
        preprocessor = SparseOneHotPreprocessor(
            numeric_columns, categorical_columns
        ).fit(train)
        estimator = self._estimator(
            n_estimators=self.max_estimators, early_stopping=True
        )
        started = time.perf_counter()
        estimator.fit(
            preprocessor.transform(train),
            y_train,
            eval_set=[(preprocessor.transform(validation), y_validation)],
            verbose=False,
        )
        duration = time.perf_counter() - started
        best_iteration = int(estimator.best_iteration) + 1
        return SelectionFit(
            estimator, preprocessor, best_iteration, duration
        )

    def refit_predict(
        self,
        selection: SelectionFit,
        train: pd.DataFrame,
        y_train: np.ndarray,
        validation: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> tuple[np.ndarray, Any, Any, float, float]:
        preprocessor = SparseOneHotPreprocessor(
            numeric_columns, categorical_columns
        ).fit(train)
        estimator = self._estimator(
            n_estimators=selection.best_iteration, early_stopping=False
        )
        started = time.perf_counter()
        estimator.fit(preprocessor.transform(train), y_train, verbose=False)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probabilities = estimator.predict_proba(
            preprocessor.transform(validation)
        )[:, 1]
        prediction_seconds = time.perf_counter() - started
        return (
            probabilities, estimator, preprocessor,
            fit_seconds, prediction_seconds,
        )


class LightGBMAdapter:
    name = 'lightgbm'
    version = lightgbm.__version__

    def __init__(self, *, max_estimators: int = 400, patience: int = 20):
        self.max_estimators = max_estimators
        self.patience = patience
        # A short matrix-layout measurement in the pilot runner freezes this.
        self.force_col_wise = True

    def _estimator(self, n_estimators: int) -> Any:
        return lightgbm.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            n_jobs=1,
            random_state=RANDOM_STATE,
            deterministic=True,
            force_col_wise=self.force_col_wise,
            force_row_wise=not self.force_col_wise,
            verbosity=-1,
        )

    def select(
        self, train: pd.DataFrame, y_train: np.ndarray,
        validation: pd.DataFrame, y_validation: np.ndarray,
        numeric_columns: list[str], categorical_columns: list[str],
    ) -> SelectionFit:
        preprocessor = StableNativeCategoryPreprocessor(
            numeric_columns, categorical_columns
        ).fit(train)
        estimator = self._estimator(self.max_estimators)
        started = time.perf_counter()
        estimator.fit(
            preprocessor.transform(train),
            y_train,
            eval_set=[(preprocessor.transform(validation), y_validation)],
            eval_metric='binary_logloss',
            categorical_feature=categorical_columns,
            callbacks=[
                lightgbm.early_stopping(self.patience, verbose=False),
                lightgbm.log_evaluation(period=0),
            ],
        )
        duration = time.perf_counter() - started
        return SelectionFit(
            estimator, preprocessor, int(estimator.best_iteration_), duration
        )

    def refit_predict(
        self, selection: SelectionFit, train: pd.DataFrame,
        y_train: np.ndarray, validation: pd.DataFrame,
        numeric_columns: list[str], categorical_columns: list[str],
    ) -> tuple[np.ndarray, Any, Any, float, float]:
        preprocessor = StableNativeCategoryPreprocessor(
            numeric_columns, categorical_columns
        ).fit(train)
        estimator = self._estimator(selection.best_iteration)
        started = time.perf_counter()
        estimator.fit(
            preprocessor.transform(train), y_train,
            categorical_feature=categorical_columns,
            callbacks=[lightgbm.log_evaluation(period=0)],
        )
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probabilities = estimator.predict_proba(
            preprocessor.transform(validation)
        )[:, 1]
        prediction_seconds = time.perf_counter() - started
        return (
            probabilities, estimator, preprocessor,
            fit_seconds, prediction_seconds,
        )


class CatBoostAdapter:
    name = 'catboost'
    version = catboost.__version__

    def __init__(self, *, max_estimators: int = 400, patience: int = 20):
        self.max_estimators = max_estimators
        self.patience = patience

    def _estimator(self, iterations: int) -> Any:
        return catboost.CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='Logloss',
            random_seed=RANDOM_STATE,
            task_type='CPU',
            thread_count=1,
            allow_writing_files=False,
            verbose=False,
        )

    def select(
        self, train: pd.DataFrame, y_train: np.ndarray,
        validation: pd.DataFrame, y_validation: np.ndarray,
        numeric_columns: list[str], categorical_columns: list[str],
    ) -> SelectionFit:
        preprocessor = CatBoostNativePreprocessor(
            numeric_columns, categorical_columns
        ).fit(train)
        train_frame = preprocessor.transform(train)
        validation_frame = preprocessor.transform(validation)
        categorical_indices = [
            train_frame.columns.get_loc(column)
            for column in categorical_columns
        ]
        estimator = self._estimator(self.max_estimators)
        started = time.perf_counter()
        estimator.fit(
            catboost.Pool(
                train_frame, y_train, cat_features=categorical_indices
            ),
            eval_set=catboost.Pool(
                validation_frame,
                y_validation,
                cat_features=categorical_indices,
            ),
            early_stopping_rounds=self.patience,
            use_best_model=True,
            verbose=False,
        )
        duration = time.perf_counter() - started
        best_iteration = int(estimator.get_best_iteration()) + 1
        return SelectionFit(
            estimator, preprocessor, best_iteration, duration
        )

    def refit_predict(
        self, selection: SelectionFit, train: pd.DataFrame,
        y_train: np.ndarray, validation: pd.DataFrame,
        numeric_columns: list[str], categorical_columns: list[str],
    ) -> tuple[np.ndarray, Any, Any, float, float]:
        preprocessor = CatBoostNativePreprocessor(
            numeric_columns, categorical_columns
        ).fit(train)
        train_frame = preprocessor.transform(train)
        validation_frame = preprocessor.transform(validation)
        categorical_indices = [
            train_frame.columns.get_loc(column)
            for column in categorical_columns
        ]
        estimator = self._estimator(selection.best_iteration)
        started = time.perf_counter()
        estimator.fit(
            catboost.Pool(
                train_frame, y_train, cat_features=categorical_indices
            ),
            verbose=False,
        )
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probabilities = estimator.predict_proba(
            catboost.Pool(
                validation_frame, cat_features=categorical_indices
            )
        )[:, 1]
        prediction_seconds = time.perf_counter() - started
        return (
            probabilities, estimator, preprocessor,
            fit_seconds, prediction_seconds,
        )


def boosting_adapters(
    *,
    max_estimators: int = 400,
    patience: int = 20,
) -> tuple[XGBoostAdapter, LightGBMAdapter, CatBoostAdapter]:
    '''Create fresh CPU-only adapters in a deterministic execution order.'''
    return (
        XGBoostAdapter(
            max_estimators=max_estimators, patience=patience
        ),
        LightGBMAdapter(
            max_estimators=max_estimators, patience=patience
        ),
        CatBoostAdapter(
            max_estimators=max_estimators, patience=patience
        ),
    )


def run_engine_on_outer_fold(
    adapter: Any,
    matrix: pd.DataFrame,
    target: np.ndarray,
    identity: pd.DataFrame,
    *,
    outer_fold_id: int,
    numeric_columns: list[str],
    categorical_columns: list[str],
    peak_rss_bytes: int,
) -> EnginePilotResult:
    '''Run exactly one internal-selection fit and one fresh outer refit.'''
    partitions = make_nested_fold_indices(
        identity, outer_fold_id=outer_fold_id
    )
    target = np.asarray(target, dtype=np.int8)
    if len(target) != len(matrix) or len(matrix) != len(identity):
        raise ValueError('Matrix, target and identity lengths must match.')
    if set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError('Pilot target must contain exactly classes 0 and 1.')
    selection = adapter.select(
        matrix.iloc[partitions.internal_train],
        target[partitions.internal_train],
        matrix.iloc[partitions.internal_validation],
        target[partitions.internal_validation],
        numeric_columns,
        categorical_columns,
    )
    (
        probabilities, final_estimator, final_preprocessor,
        refit_seconds, prediction_seconds,
    ) = adapter.refit_predict(
        selection,
        matrix.iloc[partitions.outer_train],
        target[partitions.outer_train],
        matrix.iloc[partitions.outer_validation],
        numeric_columns,
        categorical_columns,
    )
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError('Pilot probabilities must be finite.')
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError('Pilot probabilities must lie in [0, 1].')
    metrics = binary_classification_metrics(
        target[partitions.outer_validation],
        probabilities,
        threshold=CLASSIFICATION_THRESHOLD,
    )
    metrics.update(
        {
            'n_outer_train': int(len(partitions.outer_train)),
            'n_outer_validation': int(len(partitions.outer_validation)),
            'n_internal_train': int(len(partitions.internal_train)),
            'n_internal_validation': int(
                len(partitions.internal_validation)
            ),
        }
    )
    return EnginePilotResult(
        engine=adapter.name,
        outer_fold_id=partitions.outer_fold_id,
        internal_validation_fold_id=partitions.internal_validation_fold_id,
        best_iteration=selection.best_iteration,
        early_stopping_fit_seconds=selection.duration_seconds,
        outer_refit_seconds=refit_seconds,
        prediction_seconds=prediction_seconds,
        peak_rss_bytes=peak_rss_bytes,
        metrics=metrics,
        probabilities=probabilities,
        estimator_objects_are_distinct=selection.estimator is not final_estimator,
        preprocessing_objects_are_distinct=(
            selection.preprocessor is not final_preprocessor
        ),
    )
