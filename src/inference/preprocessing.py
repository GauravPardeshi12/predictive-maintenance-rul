from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import (
    remove_constant_columns,
    create_operating_condition,
    add_lag_features,
    add_rolling_features,
)
from src.utils.logger import logger


def prepare_inference_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw engine data for RUL prediction using
    the same feature engineering steps used in training.
    """

    logger.info(
        "Starting inference feature engineering."
    )

    df = df.copy()

    df = remove_constant_columns(df)

    df = create_operating_condition(df)

    df = add_lag_features(df)

    df = add_rolling_features(df)

    rows_before = len(df)

    df = df.dropna().reset_index(drop=True)

    rows_removed = rows_before - len(df)

    logger.info(
        f"Removed {rows_removed:,} rows with missing values."
    )

    logger.info(
        f"Final inference dataset shape: {df.shape}"
    )

    logger.info(
        "Inference feature engineering completed successfully."
    )

    return df