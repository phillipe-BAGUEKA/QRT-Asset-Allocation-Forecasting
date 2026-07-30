"""
Advanced boosting models for the QRT Asset Allocation Forecasting project.

This module exposes reusable factory functions for XGBoost, LightGBM and
CatBoost classification pipelines. The implementations follow the same
project conventions as the other model modules: constant-value imputation,
reproducible random seeds and Scikit-Learn compatible pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from typing import Sequence, Union

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


RANDOM_STATE = 42
VerbosityType = Union[bool, int]


def _prepare_catboost_dataframe(
    X: pd.DataFrame,
    categorical_features: tuple[str, ...],
) -> pd.DataFrame:
    """
    Prepare a pandas DataFrame for native CatBoost categorical processing.

    Numerical infinite values are converted to missing values. Categorical
    variables are converted to strings and missing categories are replaced
    by an explicit category.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.

    categorical_features : tuple[str, ...]
        Names of categorical columns handled natively by CatBoost.

    Returns
    -------
    pd.DataFrame
        Prepared copy of the input DataFrame.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame.

    ValueError
        If a requested categorical feature is missing.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Native CatBoost categorical processing requires "
            "a pandas DataFrame."
        )

    missing_features = [
        feature
        for feature in categorical_features
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing categorical features: {missing_features}."
        )

    X_prepared = X.copy()

    numerical_features = [
        column
        for column in X_prepared.columns
        if column not in categorical_features
    ]

    if numerical_features:
        X_prepared[numerical_features] = (
            X_prepared[numerical_features]
            .replace([np.inf, -np.inf], np.nan)
        )

    for feature in categorical_features:
        X_prepared[feature] = (
            X_prepared[feature]
            .astype("string")
            .fillna("__MISSING__")
            .astype(str)
        )

    return X_prepared



def build_xgboost_pipeline(
    n_estimators: int = 50,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    min_child_weight: float = 1.0,
    subsample: float = 0.7,
    colsample_bytree: float = 0.8,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Build an XGBoost classification pipeline.

    XGBoost extends classical Gradient Boosting through a regularized
    objective function and second-order optimization using gradients and
    Hessians.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting rounds.
    learning_rate : float, default=0.05
        Shrinkage applied to each tree contribution.
    max_depth : int, default=2
        Maximum tree depth.
    min_child_weight : float, default=1.0
        Minimum sum of Hessians required in a child node.
    subsample : float, default=0.7
        Fraction of observations sampled for each tree.
    colsample_bytree : float, default=0.8
        Fraction of features sampled for each tree.
    gamma : float, default=0.0
        Minimum loss reduction required to create a split.
    reg_alpha : float, default=0.0
        L1 regularization applied to leaf weights.
    reg_lambda : float, default=1.0
        L2 regularization applied to leaf weights.
    random_state : int, default=42
        Random seed used for reproducibility.
    n_jobs : int, default=-1
        Number of processor cores used during training.

    Returns
    -------
    Pipeline
        Untrained pipeline composed of ``SimpleImputer`` and
        ``XGBClassifier``.

    Raises
    ------
    ValueError
        If one of the main hyperparameters is outside its valid range.

    Notes
    -----
    Early stopping is intentionally not configured here because the current
    temporal evaluation function does not pass an ``eval_set`` to ``fit``.
    """
    if n_estimators <= 0:
        raise ValueError("n_estimators must be strictly positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be strictly positive.")
    if max_depth <= 0:
        raise ValueError("max_depth must be strictly positive.")
    if min_child_weight < 0:
        raise ValueError("min_child_weight must be non-negative.")
    if not 0 < subsample <= 1:
        raise ValueError("subsample must be in the interval (0, 1].")
    if not 0 < colsample_bytree <= 1:
        raise ValueError("colsample_bytree must be in the interval (0, 1].")
    if gamma < 0 or reg_alpha < 0 or reg_lambda < 0:
        raise ValueError(
            "gamma, reg_alpha and reg_lambda must be non-negative."
        )

    classifier = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        objective="binary:logistic",
        eval_metric="logloss",
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0),
            ),
            ("classifier", classifier),
        ]
    )


def build_lightgbm_pipeline(
    n_estimators: int = 50,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = 2,
    min_child_samples: int = 20,
    subsample: float = 0.7,
    subsample_freq: int = 1,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    verbosity: int = -1,
) -> Pipeline:
    """
    Build a LightGBM classification pipeline.

    LightGBM uses histogram-based split search and leaf-wise tree growth.
    These choices make training fast and memory efficient, but require
    explicit complexity control to limit overfitting.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting rounds.
    learning_rate : float, default=0.05
        Contribution assigned to each tree.
    num_leaves : int, default=31
        Maximum number of leaves per tree.
    max_depth : int, default=2
        Maximum tree depth. ``-1`` disables the depth limit.
    min_child_samples : int, default=20
        Minimum number of observations required in a terminal leaf.
    subsample : float, default=0.7
        Fraction of observations sampled when bagging is active.
    subsample_freq : int, default=1
        Frequency at which row subsampling is applied.
    colsample_bytree : float, default=0.8
        Fraction of variables sampled for each tree.
    reg_alpha : float, default=0.0
        L1 regularization applied to leaf scores.
    reg_lambda : float, default=0.0
        L2 regularization applied to leaf scores.
    random_state : int, default=42
        Random seed used for reproducibility.
    n_jobs : int, default=-1
        Number of processor cores used during training.
    verbosity : int, default=-1
        LightGBM logging level.

    Returns
    -------
    Pipeline
        Untrained pipeline composed of ``SimpleImputer`` and
        ``LGBMClassifier``.

    Raises
    ------
    ValueError
        If one of the main hyperparameters is outside its valid range.
    """
    if n_estimators <= 0:
        raise ValueError("n_estimators must be strictly positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be strictly positive.")
    if num_leaves <= 1:
        raise ValueError("num_leaves must be greater than 1.")
    if max_depth == 0 or max_depth < -1:
        raise ValueError("max_depth must be -1 or a strictly positive integer.")
    if min_child_samples <= 0:
        raise ValueError("min_child_samples must be strictly positive.")
    if not 0 < subsample <= 1:
        raise ValueError("subsample must be in the interval (0, 1].")
    if subsample_freq < 0:
        raise ValueError("subsample_freq must be non-negative.")
    if not 0 < colsample_bytree <= 1:
        raise ValueError("colsample_bytree must be in the interval (0, 1].")
    if reg_alpha < 0 or reg_lambda < 0:
        raise ValueError("reg_alpha and reg_lambda must be non-negative.")

    classifier = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=subsample_freq,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        objective="binary",
        verbosity=verbosity,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0),
            ),
            ("classifier", classifier),
        ]
    )


def build_catboost_categorical_pipeline(
    categorical_features: Sequence[str] = (
        "GROUP",
        "ALLOCATION",
    ),
    iterations: int = 300,
    learning_rate: float = 0.03,
    depth: int = 5,
    l2_leaf_reg: float = 8.0,
    random_strength: float = 1.0,
    random_seed: int = RANDOM_STATE,
    verbose: VerbosityType = False,
    thread_count: int = -1,
) -> Pipeline:
    """
    Build a CatBoost pipeline with native categorical feature handling.

    Parameters
    ----------
    categorical_features : sequence of str
        Columns treated natively as categorical variables.

    iterations : int, default=300
        Number of boosting iterations.

    learning_rate : float, default=0.03
        Contribution of each new tree.

    depth : int, default=5
        Depth of CatBoost symmetric trees.

    l2_leaf_reg : float, default=8.0
        L2 regularization applied to leaf values.

    random_strength : float, default=1.0
        Randomness applied during split selection.

    random_seed : int, default=42
        Random seed used for reproducibility.

    verbose : bool or int, default=False
        CatBoost logging configuration.

    thread_count : int, default=-1
        Number of CPU threads used during training.

    Returns
    -------
    Pipeline
        Scikit-Learn compatible CatBoost pipeline.
    """
    categorical_features = tuple(categorical_features)

    if not categorical_features:
        raise ValueError(
            "categorical_features must contain at least one feature."
        )

    if len(categorical_features) != len(set(categorical_features)):
        raise ValueError(
            "categorical_features contains duplicated feature names."
        )

    if iterations <= 0:
        raise ValueError("iterations must be strictly positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be strictly positive.")

    if depth <= 0:
        raise ValueError("depth must be strictly positive.")

    if l2_leaf_reg < 0:
        raise ValueError("l2_leaf_reg must be non-negative.")

    if random_strength < 0:
        raise ValueError("random_strength must be non-negative.")

    feature_preparation = FunctionTransformer(
        func=_prepare_catboost_dataframe,
        validate=False,
        kw_args={
            "categorical_features": categorical_features,
        },
    )

    classifier = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_strength=random_strength,
        cat_features=list(categorical_features),
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=random_seed,
        verbose=verbose,
        thread_count=thread_count,
        allow_writing_files=False,
    )

    return Pipeline(
        steps=[
            ("feature_preparation", feature_preparation),
            ("classifier", classifier),
        ]
    )