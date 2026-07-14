from typing import Optional, Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


MaxFeaturesType = Optional[Union[int, float, str]]


def build_decision_tree_pipeline(
    max_depth: Optional[int] = 5,
    min_samples_split: int = 2,
    min_samples_leaf: int = 20,
    max_features: MaxFeaturesType = None,
    class_weight: Optional[Union[str, dict]] = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a classification pipeline using a Decision Tree.

    The pipeline performs two consecutive operations:

    1. Missing-value imputation using a constant value of 0.
    2. Binary classification using ``DecisionTreeClassifier``.

    Feature scaling is intentionally omitted because decision trees
    split observations according to thresholds and are not sensitive
    to differences in feature scales.

    All preprocessing steps are fitted exclusively on the training
    subset of each temporal fold when the pipeline is evaluated through
    ``evaluate_model_on_folds``.

    Parameters
    ----------
    max_depth : int or None, default=5
        Maximum depth of the tree.

        A small value limits model complexity and reduces overfitting.
        If ``None``, the tree can grow until another stopping condition
        is reached.

    min_samples_split : int, default=2
        Minimum number of observations required to split an internal node.

        Increasing this value prevents the tree from creating splits
        based on very small subsets of observations.

    min_samples_leaf : int, default=20
        Minimum number of observations required in each terminal leaf.

        This parameter regularizes the tree and prevents predictions
        from being based on very small groups of observations.

    max_features : int, float, str or None, default=None
        Number of features considered when searching for the best split.

        Possible values include:

        - ``None``: all features are considered;
        - ``"sqrt"``: square root of the total number of features;
        - ``"log2"``: base-2 logarithm of the total number of features;
        - an integer: fixed number of features;
        - a float between 0 and 1: proportion of available features.

    class_weight : str, dict or None, default=None
        Weights associated with the target classes.

        ``"balanced"`` automatically adjusts class weights according
        to their frequencies in the training subset.

    random_state : int, default=42
        Random seed used to ensure reproducible results.

    Returns
    -------
    Pipeline
        An untrained Scikit-Learn pipeline composed of:

        - ``SimpleImputer``;
        - ``DecisionTreeClassifier``.
    """

    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
    )

    pipeline = Pipeline(
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

    return pipeline


def build_random_forest_pipeline(
    n_estimators: int = 200,
    max_depth: Optional[int] = 8,
    min_samples_split: int = 2,
    min_samples_leaf: int = 20,
    max_features: MaxFeaturesType = "sqrt",
    bootstrap: bool = True,
    oob_score: bool = False,
    class_weight: Optional[Union[str, dict]] = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a classification pipeline using a Random Forest.

    The pipeline performs two consecutive operations:

    1. Missing-value imputation using a constant value of 0.
    2. Binary classification using ``RandomForestClassifier``.

    A Random Forest combines multiple decision trees trained on
    bootstrap samples and introduces random feature selection during
    tree construction. This diversification reduces the correlation
    between individual trees and helps lower the variance of the final
    aggregated prediction.

    Feature scaling is intentionally omitted because tree-based models
    are not sensitive to differences in feature scales.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the forest.

        Increasing this value generally improves the stability of the
        aggregated prediction, but increases training time and memory use.

    max_depth : int or None, default=8
        Maximum depth of each tree.

        Limiting tree depth reduces the complexity and variance of the
        individual estimators.

    min_samples_split : int, default=2
        Minimum number of observations required to split an internal node.

    min_samples_leaf : int, default=20
        Minimum number of observations required in each terminal leaf.

        Increasing this value produces smoother and more regularized
        predictions.

    max_features : int, float, str or None, default="sqrt"
        Number of randomly selected features considered at each split.

        ``"sqrt"`` is a common default for classification and promotes
        diversity between trees.

    bootstrap : bool, default=True
        Whether each tree is trained on a bootstrap sample drawn with
        replacement from the training data.

    oob_score : bool, default=False
        Whether to compute the out-of-bag score.

        This score is not used as the primary evaluation criterion in
        this project because model selection relies on strict temporal
        validation.

    class_weight : str, dict or None, default=None
        Weights associated with the target classes.

        ``"balanced"`` automatically compensates for class imbalance.

    n_jobs : int, default=-1
        Number of CPU cores used during training.

        ``-1`` uses all available processors.

    random_state : int, default=42
        Random seed used to make bootstrap sampling, feature selection
        and model construction reproducible.

    Returns
    -------
    Pipeline
        An untrained Scikit-Learn pipeline composed of:

        - ``SimpleImputer``;
        - ``RandomForestClassifier``.

    Raises
    ------
    ValueError
        If ``oob_score=True`` while ``bootstrap=False``, because
        out-of-bag observations only exist when bootstrap sampling is used.
    """

    if oob_score and not bootstrap:
        raise ValueError(
            "oob_score=True requires bootstrap=True."
        )

    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        oob_score=oob_score,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )

    pipeline = Pipeline(
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

    return pipeline