"""
Cross-sectional feature engineering utilities.

This module creates features describing the relative position of an
allocation compared with the other allocations observed on the same date.
"""

import pandas as pd


REQUIRED_COLUMNS = ["TS", "RET_1"]
RET_1_PERCENTILE_COLUMN = "RET_1_percentile_by_date"


def _validate_ret1_columns(df: pd.DataFrame) -> None:
    """
    Validate the input DataFrame before creating the RET_1 percentile.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame expected to contain the columns ``TS`` and ``RET_1``.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame or if ``RET_1`` is not
        numeric.

    ValueError
        If the DataFrame is empty, required columns are missing or ``TS``
        contains missing values.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "The input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The input DataFrame is empty."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}."
        )

    if not pd.api.types.is_numeric_dtype(df["RET_1"]):
        raise TypeError(
            "The RET_1 column must be numeric."
        )

    if df["TS"].isna().any():
        raise ValueError(
            "The TS column must not contain missing values."
        )


def _validate_ret1_percentile_column(
    df: pd.DataFrame,
) -> None:
    """
    Validate the generated RET_1 intra-date percentile feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame expected to contain the generated percentile column.

    Raises
    ------
    TypeError
        If the percentile column is not numeric.

    ValueError
        If the percentile column is missing or contains values outside
        the interval ``(0, 1]``.
    """
    if RET_1_PERCENTILE_COLUMN not in df.columns:
        raise ValueError(
            f"{RET_1_PERCENTILE_COLUMN} was not created."
        )

    percentile_values = df[RET_1_PERCENTILE_COLUMN]

    if not pd.api.types.is_numeric_dtype(percentile_values):
        raise TypeError(
            f"{RET_1_PERCENTILE_COLUMN} must be numeric."
        )

    non_missing_percentiles = percentile_values.dropna()

    invalid_percentiles = (
        non_missing_percentiles.le(0.0)
        | non_missing_percentiles.gt(1.0)
    )

    if invalid_percentiles.any():
        raise ValueError(
            f"{RET_1_PERCENTILE_COLUMN} values must lie in (0, 1]."
        )


def create_return_percentile_by_date(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the cross-sectional percentile rank of RET_1 within each date.

    For each value of ``TS``, the function ranks allocations according to
    ``RET_1``. The highest recent return receives the highest percentile.

    Equal values receive their average rank. Missing ``RET_1`` values remain
    missing and are not assigned an artificial position.

    The input DataFrame is not modified.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns ``TS`` and ``RET_1``.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame enriched with the column
        ``RET_1_percentile_by_date``.

    Raises
    ------
    TypeError
        If the input is not a DataFrame or if ``RET_1`` is not numeric.

    ValueError
        If the input is empty, required columns are missing, ``TS`` contains
        missing values or the generated percentile is invalid.
    """
    _validate_ret1_columns(df)

    output_df = df.copy()

    output_df[RET_1_PERCENTILE_COLUMN] = (
        output_df
        .groupby("TS")["RET_1"]
        .rank(
            method="average",
            ascending=True,
            na_option="keep",
            pct=True,
        )
    )

    _validate_ret1_percentile_column(output_df)

    return output_df