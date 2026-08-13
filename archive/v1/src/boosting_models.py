"""
Classical boosting models for the QRT Asset Allocation Forecasting project.

This module exposes reusable factory functions for AdaBoost and
Gradient Boosting classification pipelines. Each pipeline applies the
same missing-value treatment used throughout the project and returns a
Scikit-Learn compatible estimator that can be evaluated directly with
``evaluate_model_on_folds``.
"""

from numbers import Real
from typing import Optional, Union

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


MaxFeaturesType = Optional[Union[int, float, str]]


def build_adaboost_pipeline(
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Pipeline:
    """
    Build an AdaBoost classification pipeline.

    The pipeline first imputes missing values with ``0.0`` and then fits
    an ``AdaBoostClassifier``. AdaBoost trains weak learners sequentially
    and gives more importance to observations that were previously
    misclassified.

    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of weak learners added to the ensemble.
    learning_rate : float, default=0.1
        Contribution assigned to each weak learner.
    random_state : int, default=42
        Random seed used for reproducibility.

    Returns
    -------
    Pipeline
        Untrained pipeline composed of ``SimpleImputer`` and
        ``AdaBoostClassifier``.

    Raises
    ------
    ValueError
        If ``n_estimators`` or ``learning_rate`` is not strictly positive.

    Notes
    -----
    Feature scaling is intentionally omitted because the default weak
    learners are tree-based and therefore insensitive to feature scale.
    """
    if n_estimators <= 0:
        raise ValueError("n_estimators must be strictly positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be strictly positive.")

    classifier = AdaBoostClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def build_gradboosting_pipeline(
    n_estimators: int = 50,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    min_samples_leaf: int | float = 20,
    subsample: float = 0.7,
    max_features: MaxFeaturesType = "sqrt",
    random_state: int = 42,
) -> Pipeline:
    """
    Build a Gradient Boosting classification pipeline.

    Gradient Boosting constructs an additive model in which each new tree
    reduces the loss left by the current ensemble. The implementation can
    capture nonlinear effects, thresholds and interactions between market
    variables.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of boosting stages.
    learning_rate : float, default=0.05
        Shrinkage applied to each tree contribution.
    max_depth : int, default=2
        Maximum depth of each individual regression tree.
    min_samples_leaf : int or float, default=20
        Minimum number of observations required in a terminal leaf.
    subsample : float, default=0.7
        Fraction of observations used to fit each boosting stage.
    max_features : int, float, str or None, default="sqrt"
        Number of variables considered when searching for the best split.
    random_state : int, default=42
        Random seed used for reproducibility.

    Returns
    -------
    Pipeline
        Untrained pipeline composed of ``SimpleImputer`` and
        ``GradientBoostingClassifier``.

    Raises
    ------
    ValueError
        If one of the main numerical hyperparameters is outside its valid
        range.

    Notes
    -----
    The returned pipeline is directly compatible with
    ``evaluate_model_on_folds``.
    """
    if n_estimators <= 0:
        raise ValueError("n_estimators must be strictly positive.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be strictly positive.")

    if max_depth <= 0:
        raise ValueError("max_depth must be strictly positive.")

    if isinstance(min_samples_leaf, Real) and min_samples_leaf <= 0:
        raise ValueError("min_samples_leaf must be strictly positive.")

    if not 0 < subsample <= 1:
        raise ValueError("subsample must be in the interval (0, 1].")

    classifier = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=0.0,
                ),
            ),
            ("classifier", classifier),
        ]
    )
