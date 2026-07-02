import numpy as np
import pandas as pd

def baseline_majority_class(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict the majority class observed in the current training fold.

    The function identifies the most frequent value of the binary
    target column ``class`` in the training DataFrame and assigns this
    class to every observation of the validation DataFrame.

    This strategy ignores all explanatory variables and provides a
    minimum reference performance for subsequent prediction methods.

    Args:
        df_train:
            Training observations of the current temporal fold.
            The DataFrame must contain the binary target column
            ``class``.

        df_validation:
            Validation observations of the current temporal fold.

    Returns:
        A copy of ``df_validation`` with an additional integer column
        named ``prediction`` containing the majority class learned
        from ``df_train``.

    Notes:
        Common input validations, such as checking for empty DataFrames
        and required evaluation columns, are performed by the temporal
        evaluation function.
    """

    df_validation = df_validation.copy()
    prediction = df_train["class"].value_counts().idxmax()
    df_validation["prediction"] = prediction
    return df_validation


def baseline_always_0(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict class 0 for every validation observation.

    This constant baseline ignores both the explanatory variables and
    the class distribution observed in the training fold. It always
    predicts the negative class.

    Args:
        df_train:
            Training observations of the current temporal fold.
            This argument is not used by the prediction rule but is kept
            to preserve a common interface across all baseline functions.

        df_validation:
            Validation observations of the current temporal fold.

    Returns:
        A copy of ``df_validation`` with an additional integer column
        named ``prediction`` filled with class 0.

    Notes:
        Common input validations are performed by the temporal
        evaluation function.
    """

    df_validation = df_validation.copy()
    df_validation["prediction"] = 0
    return df_validation


def baseline_always_1(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict class 1 for every validation observation.

    This constant baseline ignores both the explanatory variables and
    the class distribution observed in the training fold. It always
    predicts the positive class.

    Args:
        df_train:
            Training observations of the current temporal fold.
            This argument is not used by the prediction rule but is kept
            to preserve a common interface across all baseline functions.

        df_validation:
            Validation observations of the current temporal fold.

    Returns:
        A copy of ``df_validation`` with an additional integer column
        named ``prediction`` filled with class 1.

    Notes:
        Common input validations are performed by the temporal
        evaluation function.
    """

    df_validation = df_validation.copy()
    df_validation["prediction"] = 1
    return df_validation


def baseline_momentum(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict the future class using a one-period momentum rule.

    The prediction follows the sign of the most recent observed return:

    - ``RET_1 > 0`` produces class 1;
    - ``RET_1 <= 0`` produces class 0.

    The rule tests whether a positive recent return tends to be followed
    by a positive future return.

    Args:
        df_train:
            Training observations of the current temporal fold.
            This argument is not used by the prediction rule but is kept
            to preserve a common interface across baseline functions.

        df_validation:
            Validation observations containing the ``RET_1`` column.

    Returns:
        A copy of ``df_validation`` with an additional integer column
        named ``prediction``.

    Raises:
        ValueError:
            If the required column ``RET_1`` is missing from
            ``df_validation``.

    Notes:
        A missing value in ``RET_1`` does not satisfy ``RET_1 > 0`` and
        therefore produces class 0 with the current implementation.
    """

    df_validation = df_validation.copy()
    if "RET_1" not in df_validation.columns:
        raise ValueError("Missing required column: RET_1.")
    else :
        df_validation["prediction"] = np.where(df_validation["RET_1"] > 0,1,0)
    
    return df_validation

def baseline_reversal(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict the future class using a one-period reversal rule.

    The rule predicts a positive future class only when the most recent
    observed return is strictly negative:

    - ``RET_1 < 0`` produces class 1;
    - ``RET_1 >= 0`` produces class 0.

    This strategy tests whether a negative recent return tends to be
    followed by a positive correction.

    Args:
        df_train:
            Training observations of the current temporal fold.
            This argument is not used by the prediction rule but is kept
            to preserve a common interface across baseline functions.

        df_validation:
            Validation observations containing the ``RET_1`` column.

    Returns:
        A copy of ``df_validation`` with an additional integer column
        named ``prediction``.

    Raises:
        ValueError:
            If the required column ``RET_1`` is missing from
            ``df_validation``.

    Notes:
        A missing value in ``RET_1`` does not satisfy ``RET_1 < 0`` and
        therefore produces class 0 with the current implementation.
    """

    df_validation = df_validation.copy()
    if "RET_1" not in df_validation.columns:
        raise ValueError("Missing required column: RET_1.")
    else :
        df_validation["prediction"] = np.where(df_validation["RET_1"] < 0 ,1,0)
    
    return df_validation


def baseline_momentum_agr(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict the future class using an aggregated momentum rule.

    The function computes the row-wise mean of the three most recent
    returns: ``RET_1``, ``RET_2`` and ``RET_3``.

    The prediction rule is:

    - aggregated return greater than 0 produces class 1;
    - aggregated return lower than or equal to 0 produces class 0.

    This strategy tests whether the average direction of several recent
    returns contains a more stable momentum signal than ``RET_1`` alone.

    Args:
        df_train:
            Training observations of the current temporal fold.
            This argument is not used by the prediction rule but is kept
            to preserve a common interface across baseline functions.

        df_validation:
            Validation observations containing ``RET_1``, ``RET_2`` and
            ``RET_3``.

    Returns:
        A copy of ``df_validation`` with an additional integer column
        named ``prediction``.

    Raises:
        ValueError:
            If at least one of the required return columns is missing.

    Notes:
        Pandas ignores available missing values when computing the
        row-wise mean. The aggregated return may therefore be calculated
        from fewer than three available returns. If all three values are
        missing, the resulting prediction is class 0.
    """

    df_validation = df_validation.copy()

    required_cols = ["RET_1", "RET_2", "RET_3"]

    if not all(col in df_validation.columns for col in required_cols):
        raise ValueError("Missing required columns: RET_1, RET_2, RET_3.")

    aggregated_return = df_validation[required_cols].mean(axis=1)

    df_validation["prediction"] = np.where(aggregated_return > 0, 1, 0)

    return df_validation


def baseline_reversal_agr(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict the future class using an aggregated reversal rule.

    The function computes the row-wise mean of the three most recent
    returns: ``RET_1``, ``RET_2`` and ``RET_3``.

    The prediction rule is:

    - aggregated return lower than 0 produces class 1;
    - aggregated return greater than or equal to 0 produces class 0.

    This strategy tests whether a negative average recent return tends
    to be followed by a positive correction.

    Args:
        df_train:
            Training observations of the current temporal fold.
            This argument is not used by the prediction rule but is kept
            to preserve a common interface across baseline functions.

        df_validation:
            Validation observations containing ``RET_1``, ``RET_2`` and
            ``RET_3``.

    Returns:
        A copy of ``df_validation`` with an additional integer column
        named ``prediction``.

    Raises:
        ValueError:
            If at least one of the required return columns is missing.

    Notes:
        Pandas ignores available missing values when computing the
        row-wise mean. The aggregated return may therefore be calculated
        from fewer than three available returns. If all three values are
        missing, the resulting prediction is class 0.
    """

    df_validation = df_validation.copy()

    required_cols = ["RET_1", "RET_2", "RET_3"]

    if not all(col in df_validation.columns for col in required_cols):
        raise ValueError("Missing required columns: RET_1, RET_2, RET_3.")

    aggregated_return = df_validation[required_cols].mean(axis=1)

    df_validation["prediction"] = np.where(aggregated_return < 0 ,1,0)

    return df_validation

