"""
Advanced boosting models for the QRT Asset Allocation Forecasting project.

This module exposes reusable factory functions for XGBoost, LightGBM and
CatBoost classification pipelines. The implementations follow the same
project conventions as the other model modules: constant-value imputation,
reproducible random seeds and Scikit-Learn compatible pipelines.
"""

from typing import Union

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


RANDOM_STATE = 42
VerbosityType = Union[bool, int]


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


def build_catboost_pipeline(
    iterations: int = 100,
    learning_rate: float = 0.1,
    depth: int = 6,
    l2_leaf_reg: float = 3.0,
    random_seed: int = RANDOM_STATE,
    verbose: VerbosityType = False,
    thread_count: int = -1,
) -> Pipeline:
    """
    Build a CatBoost classification pipeline.

    CatBoost relies on ordered boosting and symmetric trees. It is
    especially relevant when categorical features are handled natively,
    although this project version currently receives numerical features.

    Parameters
    ----------
    iterations : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Contribution assigned to each tree.
    depth : int, default=6
        Depth of the symmetric trees.
    l2_leaf_reg : float, default=3.0
        L2 regularization coefficient applied to leaf values.
    random_seed : int, default=42
        Random seed used for reproducibility.
    verbose : bool or int, default=False
        Logging configuration.
    thread_count : int, default=-1
        Number of processor threads used during training.

    Returns
    -------
    Pipeline
        Untrained pipeline composed of ``SimpleImputer`` and
        ``CatBoostClassifier``.

    Raises
    ------
    ValueError
        If one of the main hyperparameters is outside its valid range.
    """
    if iterations <= 0:
        raise ValueError("iterations must be strictly positive.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be strictly positive.")
    if depth <= 0:
        raise ValueError("depth must be strictly positive.")
    if l2_leaf_reg < 0:
        raise ValueError("l2_leaf_reg must be non-negative.")

    classifier = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        verbose=verbose,
        thread_count=thread_count,
        loss_function="Logloss",
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
