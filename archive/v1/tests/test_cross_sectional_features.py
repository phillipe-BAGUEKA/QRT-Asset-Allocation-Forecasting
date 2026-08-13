"""
Unit tests for cross-sectional feature engineering utilities.

These tests verify that invalid inputs are correctly rejected by
create_return_percentile_by_date.
"""

import numpy as np
import pandas as pd
import pytest

from src.cross_sectional_features import (
    create_return_percentile_by_date,
)


def test_non_numeric_ret1_raises_type_error() -> None:
    """
    Verify that a non-numeric RET_1 column raises a TypeError.
    """
    # Arrange
    input_df = pd.DataFrame(
        {
            "RET_1": ["0.3", 2, 4],
            "TS": ["001", "002", "003"],
        }
    )

    # Act + Assert
    with pytest.raises(
        TypeError,
        match=r"The RET_1 column must be numeric\.",
    ):
        create_return_percentile_by_date(input_df)


def test_empty_dataframe_raises_value_error() -> None:
    """
    Verify that an empty DataFrame raises a ValueError.
    """
    # Arrange
    input_df = pd.DataFrame()

    # Act + Assert
    with pytest.raises(
        ValueError,
        match=r"The input DataFrame is empty\.",
    ):
        create_return_percentile_by_date(input_df)


def test_missing_required_column_raises_value_error() -> None:
    """
    Verify that a missing required column raises a ValueError.
    """
    # Arrange
    input_df = pd.DataFrame(
        {
            "RET": [0.3, 2.0, 4.0],
            "TS": ["001", "002", "003"],
        }
    )

    # Act + Assert
    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        create_return_percentile_by_date(input_df)


def test_missing_ts_value_raises_value_error() -> None:
    """
    Verify that missing values in TS raise a ValueError.
    """
    # Arrange
    input_df = pd.DataFrame(
        {
            "RET_1": [1.0, 2.0, 4.0],
            "TS": ["001", np.nan, "003"],
        }
    )

    # Act + Assert
    with pytest.raises(
        ValueError,
        match=r"The TS column must not contain missing values\.",
    ):
        create_return_percentile_by_date(input_df)

# Test 1 — Percentiles exacts sur une date

def test_percentiles_match_expected_values() -> None:
    # Arrange
    input_df = pd.DataFrame(
        {
            "RET_1": [0.03, 0.02, 1.0, 0.03, 0.06],
            "TS": ["001", "001", "001", "002", "002"],
        }
    )

    expected_percentiles = [
        2 / 3,
        1 / 3,
        1.0,
        1 / 2,
        1.0,
    ]

    # Act
    output_df = create_return_percentile_by_date(input_df)

    observed_percentiles = output_df[
        "RET_1_percentile_by_date"
    ].tolist()

    # Assert
    assert observed_percentiles == pytest.approx(
        expected_percentiles
    )

def test_equal_returns_receive_equal_average_rank() -> None:
    # Arrange
    input_df = pd.DataFrame({
        "RET_1": [0.02, 0.03, 0.03, 0.04],
        "TS": ["001", "001", "001", "001"]
    })

    expected_percentiles = [
        1 / 4,
        2.5 / 4,
        2.5 / 4,
        1
    ]

    # Act
    output_df = create_return_percentile_by_date(input_df)

    observed_percentiles = output_df[
        "RET_1_percentile_by_date"
    ].tolist() 

    # Assert 
    assert observed_percentiles == pytest.approx(
        expected_percentiles
    )

def test_missing_ret1_keeps_missing_percentile() -> None:
    # Arrange
    input_df = pd.DataFrame(
        {
            "RET_1": [0.02, np.nan, 0.03, np.nan],
            "TS": ["001", "001", "001", "001"],
        }
    )

    expected_percentiles = [
        0.5,
        np.nan,
        1.0,
        np.nan,
    ]

    # Act
    output_df = create_return_percentile_by_date(input_df)

    observed_percentiles = output_df[
        "RET_1_percentile_by_date"
    ].tolist()

    # Assert
    assert observed_percentiles[::2] == pytest.approx(
        expected_percentiles[::2]
    )

    assert pd.isna(observed_percentiles[1::2]).all()


def test_input_dataframe_is_not_modified_and_row_order_is_preserved() -> None:
    # Arrange
    input_df = pd.DataFrame(
        {
            "ROW_ID": [1, 2, 3],
            "TS": ["001", "001", "001"],
            "RET_1": [0.02, 0.03, 0.04],
        },
        index=[1, 2, 3],
    )

    original_df = input_df.copy(deep=True)
    returned_column = "RET_1_percentile_by_date"

    # Act
    output_df = create_return_percentile_by_date(input_df)

    # Assert
    assert returned_column in output_df.columns
    assert returned_column not in input_df.columns

    pd.testing.assert_frame_equal(
        input_df,
        original_df,
    )

    assert len(input_df) == len(output_df)
    assert input_df["ROW_ID"].equals(output_df["ROW_ID"])
    assert input_df.index.equals(output_df.index)

    assert output_df is not input_df







