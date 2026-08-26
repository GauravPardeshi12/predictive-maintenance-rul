from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import (
    remove_constant_columns,
    create_operating_condition_labels,
    create_lag_features,
    create_rolling_features,
)
from src.utils.logger import logger


def prepare_inference_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw engine data for RUL prediction.

    The same feature engineering steps used during
    training are applied, except RUL calculation is
    not performed because RUL is the prediction target.
    """

    logger.info(
        "Starting inference feature engineering."
    )

    df = df.copy()

    df = remove_constant_columns(df)

    df = create_operating_condition_labels(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    logger.info(
        f"Inference feature engineering completed. "
        f"Final shape: {df.shape}"
    )

    return df