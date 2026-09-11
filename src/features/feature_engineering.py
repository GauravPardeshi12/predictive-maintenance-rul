from __future__ import annotations

import pandas as pd

from src.utils.config import config
from src.utils.dataset_utils import get_sensor_columns
from src.utils.logger import logger


WINDOW_SIZE = config["feature_engineering"]["rolling_window"]
GROUP_COLUMNS = ["dataset_id", "unit_id"]


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns containing a single unique value."""
    constant_columns = [column for column in df.columns if df[column].nunique() <= 1]

    if constant_columns:
        df = df.drop(columns=constant_columns)
        logger.info(f"Removed {len(constant_columns)} constant columns")

    return df


def create_operating_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact operating-condition identifier from the settings."""
    for column in ("setting_1", "setting_2", "setting_3"):
        df[column] = df[column].round(1)

    df["operating_condition"] = (
        df["setting_1"].astype(str)
        + "_"
        + df["setting_2"].astype(str)
        + "_"
        + df["setting_3"].astype(str)
    )

    logger.info(
        f"Identified {df['operating_condition'].nunique()} operating conditions"
    )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add one-cycle lag features for all sensor columns."""
    sensor_columns = get_sensor_columns()
    lagged = df.groupby(GROUP_COLUMNS, sort=False)[sensor_columns].shift(1)
    lagged.columns = [f"{column}_lag1" for column in sensor_columns]
    df[lagged.columns] = lagged
    logger.info(f"Created {len(sensor_columns)} lag features")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean features for all sensor columns."""
    sensor_columns = get_sensor_columns()
    rolling = (
        df.groupby(GROUP_COLUMNS, sort=False)[sensor_columns]
        .rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE)
        .mean()
        .reset_index(level=GROUP_COLUMNS, drop=True)
    )
    rolling.columns = [
        f"{column}_rolling_mean_{WINDOW_SIZE}" for column in sensor_columns
    ]
    df[rolling.columns] = rolling
    logger.info(f"Created {len(sensor_columns)} rolling-mean features")
    return df


def prepare_features(
    df: pd.DataFrame,
    remove_constants: bool = True,
) -> pd.DataFrame:
    """Run the common feature-engineering pipeline used by train and inference."""
    logger.info("Starting feature engineering pipeline")

    df = df.copy()
    df = df.sort_values(GROUP_COLUMNS + ["cycle"]).reset_index(drop=True)

    if remove_constants:
        df = remove_constant_columns(df)

    df = create_operating_condition(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)

    logger.info(f"Removed {rows_before - len(df):,} rows created by lag/rolling features")
    logger.info(f"Final feature-engineered shape: {df.shape}")
    return df


def prepare_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the training dataframe."""
    return prepare_features(df, remove_constants=True)
