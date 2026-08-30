import pandas as pd


MOMENTUM_WINDOWS = [3, 5, 10, 20]
VOLATILITY_WINDOWS = [3, 20]
VOLUME_WINDOWS = [3, 20]


def _validate_ret_columns(df: pd.DataFrame) -> None:
    """
    Validate that all required historical return columns are present.
    """
    missing_ret_cols = [
        f"RET_{i}"
        for i in range(1, 21)
        if f"RET_{i}" not in df.columns
    ]

    if missing_ret_cols:
        raise ValueError(f"Missing RET columns: {missing_ret_cols}")


def _validate_signed_volume_columns(df: pd.DataFrame) -> None:
    """
    Validate that all required historical signed volume columns are present.
    """
    missing_volume_cols = [
        f"SIGNED_VOLUME_{i}"
        for i in range(1, 21)
        if f"SIGNED_VOLUME_{i}" not in df.columns
    ]

    if missing_volume_cols:
        raise ValueError(f"Missing SIGNED_VOLUME columns: {missing_volume_cols}")


def _validate_momentum_features(df: pd.DataFrame) -> None:
    """
    Validate that all momentum features have been successfully created.
    """
    missing_momentum_features = [
        f"momentum_{window}"
        for window in MOMENTUM_WINDOWS
        if f"momentum_{window}" not in df.columns
    ]

    if missing_momentum_features:
        raise ValueError(
            f"Missing momentum columns: {missing_momentum_features}"
        )


def _validate_delta_momentum_features(df: pd.DataFrame) -> None:
    """
    Validate that the delta momentum feature exists.
    """
    if "delta_momentum_3_20" not in df.columns:
        raise ValueError("Missing column: delta_momentum_3_20")


def _validate_volatility_features(df: pd.DataFrame) -> None:
    """
    Validate that all volatility features have been successfully created.
    """
    missing_volatility_features = [
        f"volatility_{window}"
        for window in VOLATILITY_WINDOWS
        if f"volatility_{window}" not in df.columns
    ]

    if missing_volatility_features:
        raise ValueError(
            f"Missing volatility columns: {missing_volatility_features}"
        )


def _validate_delta_volatility_features(df: pd.DataFrame) -> None:
    """
    Validate that the delta volatility feature exists.
    """
    if "delta_volatility_3_20" not in df.columns:
        raise ValueError("Missing column: delta_volatility_3_20")


def _validate_momentum_volatility_ratio_features(df: pd.DataFrame) -> None:
    """
    Validate that the momentum-volatility ratio feature exists.
    """
    if "momentum_volatility_ratio_20" not in df.columns:
        raise ValueError("Missing column: momentum_volatility_ratio_20")


def _validate_volume_features(df: pd.DataFrame) -> None:
    """
    Validate that all volume features have been successfully created.
    """
    missing_volume_features = [
        f"volume_{window}"
        for window in VOLUME_WINDOWS
        if f"volume_{window}" not in df.columns
    ]

    if missing_volume_features:
        raise ValueError(
            f"Missing volume columns: {missing_volume_features}"
        )


def _validate_delta_volume_features(df: pd.DataFrame) -> None:
    """
    Validate that the delta volume feature exists.
    """
    if "delta_volume_3_20" not in df.columns:
        raise ValueError("Missing column: delta_volume_3_20")


def create_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create multi-horizon momentum features from historical returns.

    The function computes the arithmetic mean of historical returns
    over several temporal horizons in order to summarize the recent
    market direction.

    Returns a copy of the input DataFrame enriched with:
    ``momentum_3``, ``momentum_5``, ``momentum_10``, ``momentum_20``,
    and ``delta_momentum_3_20``.
    """
    df = df.copy()

    _validate_ret_columns(df)

    for window in MOMENTUM_WINDOWS:
        return_cols = [f"RET_{i}" for i in range(1, window + 1)]
        df[f"momentum_{window}"] = df[return_cols].mean(axis=1)

    _validate_momentum_features(df)

    df["delta_momentum_3_20"] = (
        df["momentum_3"] - df["momentum_20"]
    )

    _validate_delta_momentum_features(df)

    return df


def create_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create multi-horizon realized volatility features from historical returns.

    The function computes the row-wise standard deviation of historical
    returns over short-term and long-term horizons.

    Returns a copy of the input DataFrame enriched with:
    ``volatility_3``, ``volatility_20``,
    and ``delta_volatility_3_20``.
    """
    df = df.copy()

    _validate_ret_columns(df)

    for window in VOLATILITY_WINDOWS:
        return_cols = [f"RET_{i}" for i in range(1, window + 1)]
        df[f"volatility_{window}"] = df[return_cols].std(axis=1)

    _validate_volatility_features(df)

    df["delta_volatility_3_20"] = (
        df["volatility_3"] - df["volatility_20"]
    )

    _validate_delta_volatility_features(df)

    return df


def create_momentum_volatility_ratio_features(
    df: pd.DataFrame,
    epsilon: float = 1e-4,
) -> pd.DataFrame:
    """
    Create a risk-adjusted momentum feature.

    The function computes the ratio between long-term momentum
    and long-term realized volatility:

    ``momentum_volatility_ratio_20 = momentum_20 / (volatility_20 + epsilon)``

    This feature measures the quality of the directional signal,
    or how much average return is obtained per unit of historical noise.
    """
    df = df.copy()

    _validate_momentum_features(df)
    _validate_volatility_features(df)

    df["momentum_volatility_ratio_20"] = (
        df["momentum_20"] / (df["volatility_20"] + epsilon)
    )

    _validate_momentum_volatility_ratio_features(df)

    return df


def create_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create multi-horizon signed volume features.

    The function computes the arithmetic mean of signed volumes
    over short-term and long-term horizons in order to summarize
    the recent level of market activity associated with each allocation.

    Returns a copy of the input DataFrame enriched with:
    ``volume_3``, ``volume_20`` and ``delta_volume_3_20``.
    """
    df = df.copy()

    _validate_signed_volume_columns(df)

    for window in VOLUME_WINDOWS:
        volume_cols = [
            f"SIGNED_VOLUME_{i}"
            for i in range(1, window + 1)
        ]

        df[f"volume_{window}"] = df[volume_cols].mean(axis=1)

    _validate_volume_features(df)

    df["delta_volume_3_20"] = (
        df["volume_3"] - df["volume_20"]
    )

    _validate_delta_volume_features(df)

    return df





def ret_features(df:pd.DataFrame) -> list[str]:
    _validate_ret_columns(df)
    
    ret_features = [
        f"RET_{i}"
        for i in range(1,21)
    ]

    return ret_features