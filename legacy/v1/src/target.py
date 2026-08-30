import numpy as np
import pandas as pd


def _validate_merge_columns(X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
    """
    Validate that the required columns are available to build the target.

    The function verifies that both training DataFrames contain
    the columns required to merge the observations and create
    the binary classification target.

    Args:
        X_train:
            Feature DataFrame expected to contain ``ROW_ID``.

        y_train:
            Target DataFrame expected to contain both
            ``ROW_ID`` and ``target``.

    Raises:
        ValueError:
            If one or more required columns are missing.
    """

    if "ROW_ID" not in X_train.columns or "ROW_ID" not in y_train.columns:
        raise ValueError("ROW_ID is required to merge X_train and y_train.")

    if "target" not in y_train.columns:
        raise ValueError("target is required to create the class column.")


def _validate_class_column(df: pd.DataFrame) -> None:
    """
    Validate that the binary class column has been created.

    The function checks that the engineered ``class`` column
    is present after converting the regression target into
    a binary classification target.

    Args:
        df:
            DataFrame expected to contain the ``class`` column.

    Raises:
        ValueError:
            If the class column is missing.
    """

    if "class" not in df.columns:
        raise ValueError("The class column was not created.")


def create_class_column(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the binary classification target.

    The function converts the continuous target into
    a binary variable:

    - class = 1 if target > 0
    - class = 0 otherwise

    The resulting target is then merged with the
    training feature set using ``ROW_ID``.

    Args:
        X_train:
            Training feature DataFrame.

        y_train:
            Training target DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:

            - Training DataFrame enriched with the binary class.
            - Updated target DataFrame containing the ``class`` column.

    Raises:
        ValueError:
            If the merge columns are missing or if the class
            column could not be created.
    """
    
    X_train = X_train.copy()
    y_train = y_train.copy()

    _validate_merge_columns(X_train, y_train)

    y_train["class"] = np.where(y_train["target"] > 0, 1, 0)

    df_train = X_train.merge(
        y_train,
        on="ROW_ID",
        how="inner",
    )

    _validate_class_column(df_train)

    return df_train, y_train