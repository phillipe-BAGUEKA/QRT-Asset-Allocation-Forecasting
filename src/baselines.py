import numpy as np
import pandas as pd

def baseline_majority_class(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict the majority class observed in the training fold.

    This baseline ignores all explanatory variables and predicts
    the most frequent class in the training set for every observation
    of the validation fold.

    Args:
        df_train: Training observations of the current fold.
        df_validation: Validation observations of the current fold.

    Returns:
        Validation DataFrame with an additional 'prediction' column.

    Raises:
        ValueError: If the input DataFrames are empty or if the target
        column is missing.
    """

    df_train = df_train.copy()
    df_validation = df_validation.copy()
    if (df_train.empty == True) or (df_validation.empty == True) :
        raise ValueError("Le set de validation ou de train est vide !")
    if "class" not in df_train.columns:
        raise ValueError("La colonne class est manquante !")
    else :
        prediction = df_train["class"].value_counts().idxmax()
        df_validation["prediction"] = prediction
    return df_validation


def baseline_always_0(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict class 0 for every validation observation.

    This baseline completely ignores the training data and always
    predicts the negative class.

    Args:
        df_train: Training observations of the current fold.
        df_validation: Validation observations of the current fold.

    Returns:
        Validation DataFrame with an additional 'prediction' column.

    Raises:
        ValueError: If the input DataFrames are empty or if the target
        column is missing.
    """

    df_train = df_train.copy()
    df_validation = df_validation.copy()
    if (df_train.empty == True) or (df_validation.empty == True) :
        raise ValueError("Le set de validation ou de train est vide !")
    if "class" not in df_train.columns:
        raise ValueError("La colonne class est manquante !")
    else :
        df_validation["prediction"] = 0
    return df_validation


def baseline_always_1(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict class 1 for every validation observation.

    This baseline completely ignores the training data and always
    predicts the positive class.

    Args:
        df_train: Training observations of the current fold.
        df_validation: Validation observations of the current fold.

    Returns:
        Validation DataFrame with an additional 'prediction' column.

    Raises:
        ValueError: If the input DataFrames are empty or if the target
        column is missing.
    """

    df_train = df_train.copy()
    df_validation = df_validation.copy()
    if (df_train.empty == True) or (df_validation.empty == True) :
        raise ValueError("Le set de validation ou de train est vide !")
    if "class" not in df_train.columns:
        raise ValueError("La colonne class est manquante !")
    else :
        df_validation["prediction"] = 1
    return df_validation


def baseline_momentum(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict the future class using a simple momentum rule.

    The prediction follows the sign of the most recent return (RET_1):
        - RET_1 > 0 -> class 1
        - otherwise -> class 0

    Args:
        df_train: Training observations of the current fold.
        df_validation: Validation observations of the current fold.

    Returns:
        Validation DataFrame with an additional 'prediction' column.

    Raises:
        ValueError: If the input DataFrames are empty or if RET_1
        is missing.
    """

    df_train = df_train.copy()
    df_validation = df_validation.copy()
    if (df_train.empty == True) or (df_validation.empty == True) :
        raise ValueError("Le set de validation ou de train est vide !")
    if "RET_1" not in df_validation.columns:
        raise ValueError("La colonne RET_1 est manquante !")
    else :
        df_validation["prediction"] = np.where(df_validation["RET_1"] > 0,1,0)
    
    return df_validation

def baseline_reversal(df_train:pd.DataFrame,df_validation:pd.DataFrame) -> pd.DataFrame:
    """
    Predict the future class using a simple reversal rule.

    The prediction is opposite to the sign of the most recent return
    (RET_1):
        - RET_1 > 0 -> class 0
        - otherwise -> class 1

    Args:
        df_train: Training observations of the current fold.
        df_validation: Validation observations of the current fold.

    Returns:
        Validation DataFrame with an additional 'prediction' column.

    Raises:
        ValueError: If the input DataFrames are empty or if RET_1
        is missing.
    """
    df_train = df_train.copy()
    df_validation = df_validation.copy()
    if (df_train.empty == True) or (df_validation.empty == True) :
        raise ValueError("Le set de validation ou de train est vide !")
    if "RET_1" not in df_validation.columns:
        raise ValueError("La colonne RET_1 est manquante !")
    else :
        df_validation["prediction"] = np.where(df_validation["RET_1"] > 0,0,1)
    
    return df_validation


def baseline_momentum_agr(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict the future class using an aggregated momentum rule.

    The prediction is based on the average of the three most recent
    returns (RET_1, RET_2 and RET_3):
        - average return > 0 -> class 1
        - otherwise -> class 0

    Args:
        df_train: Training observations of the current fold.
        df_validation: Validation observations of the current fold.

    Returns:
        Validation DataFrame with an additional 'prediction' column.

    Raises:
        ValueError: If the input DataFrames are empty or if one of the
        required return columns is missing.
    """

    df_train = df_train.copy()
    df_validation = df_validation.copy()

    if df_train.empty or df_validation.empty:
        raise ValueError("Le set de validation ou de train est vide !")

    required_cols = ["RET_1", "RET_2", "RET_3"]

    if not all(col in df_validation.columns for col in required_cols):
        raise ValueError("Les colonnes de rendements récents sont manquantes !")

    aggregated_return = df_validation[required_cols].mean(axis=1)

    df_validation["prediction"] = np.where(aggregated_return > 0, 1, 0)

    return df_validation


def baseline_reversal_agr(
    df_train: pd.DataFrame,
    df_validation: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict the future class using an aggregated reversal rule.

    The prediction is opposite to the average of the three most recent
    returns (RET_1, RET_2 and RET_3):
        - average return > 0 -> class 0
        - otherwise -> class 1

    Args:
        df_train: Training observations of the current fold.
        df_validation: Validation observations of the current fold.

    Returns:
        Validation DataFrame with an additional 'prediction' column.

    Raises:
        ValueError: If the input DataFrames are empty or if one of the
        required return columns is missing.
    """

    df_train = df_train.copy()
    df_validation = df_validation.copy()

    if df_train.empty or df_validation.empty:
        raise ValueError("Le set de validation ou de train est vide !")

    required_cols = ["RET_1", "RET_2", "RET_3"]

    if not all(col in df_validation.columns for col in required_cols):
        raise ValueError("Les colonnes de rendements récents sont manquantes !")

    aggregated_return = df_validation[required_cols].mean(axis=1)

    df_validation["prediction"] = np.where(aggregated_return > 0, 0, 1)

    return df_validation

