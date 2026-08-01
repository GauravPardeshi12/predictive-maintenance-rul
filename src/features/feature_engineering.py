from __future__ import annotations
import pandas as pd
from src.utils.logger import logger
from src.utils.config import config
from src.utils.dataset_utils import get_sensor_columns

NUM_SENSORS = config["dataset"]["num_sensors"]
WINDOW_SIZE = config["feature_engineering"]["rolling_window"]


def remove_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that contains only one unique value

    Parameters
    ----------
    pd.DataFrame
        Input dataset

    Returns
    ------------
    pd.DataFrame
        dataset with constant columns removed
    """
    logger.info("Removing constant columns.")

    constant_columns = [column for column in df.columns if df[column].nunique() == 1]

    df = df.drop(columns=constant_columns)

    if constant_columns:
        logger.info(f"Removed {len(constant_columns)} constant columns.")
    else:
        logger.info("No constant columns found.")

    return df

def create_operating_condition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an operating condition identifier

    Parameters
    -----------
    df: pd.DataFrame
        Input Dataset

    Returns
    ------------
    df: pd.DataFrame
        Dataset with operating condition Labels.
    """
    logger.info("Creating operating condition labels.")

    df["setting_1"] = df["setting_1"].round(1)
    df["setting_2"] = df["setting_2"].round(1)
    df["setting_3"] = df["setting_3"].round(1)

    df["operating_condition"] = df["setting_1"].astype(str) + "_" + df["setting_2"].astype(str) + "_" + df["setting_3"].astype(str)

    logger.info(f"Identified {df['operating_condition'].nunique()} operating conditions")

    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Lag features for selected sensor columns.

    Parameters
    ----------
    df: pd.DataFrame
        Input Dataset

    Returns
    -----------
    df: pd.DataFrame
        Dataset with lag features.
    """
    logger.info("Creating lag features.")

    sensor_columns = get_sensor_columns()

    for column in sensor_columns:
        df[f"{column}_lag1"] = df.groupby(["dataset_id", "unit_id"])[column].shift(1)

    logger.info(f"Created lag features for {len(sensor_columns)} sensors")

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    create rolling statistics for sensor columns.

    Parameters
    ----------
    df: pd.DataFrame
        Input dataset

    Returns
    df: pd.DataFrame
        Dataset with rolling features
    """
    logger.info("Creating rolling features.")

    sensor_columns = get_sensor_columns()

    for column in sensor_columns:
        df[f"{column}_rolling_mean_{WINDOW_SIZE}"] = (
            df.groupby(["dataset_id", "unit_id"])[column]
            .rolling(window=WINDOW_SIZE)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    logger.info(f"Created rolling mean features for {len(sensor_columns)} sensors.")

    return df


def prepare_training_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the training dataset by applying feature engineering steps

    Parameters
    ------------
    df: pd.DataFrame
        Input dataset

    Returns
    -----------
    df: pd.DataFrame
        Processed training dataset
    """
    logger.info("Starting feature engineering pipeline.")

    df = remove_constant_columns(df)

    df = create_operating_condition(df)

    df = add_lag_features(df)

    df = add_rolling_features(df)

    logger.info("Removing rows with missing values.")

    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_removed = rows_before - len(df)

    logger.info(f"Removed {rows_removed} rows with missing values.")

    logger.info(f"Final dataset shape: {df.shape}")

    logger.info("Feature engineering pipeline completed successfully.")
    return df
