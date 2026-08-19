from __future__ import annotations

import pandas as pd

from xgboost import XGBRegressor

from src.utils.logger import logger


def train_optimized_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> XGBRegressor:
    """
    Train the final XGBoost model using the hyperparameters
    selected by Optuna.

    Parameters
    ----------
    x_train : pd.DataFrame
        Complete training feature matrix.

    y_train : pd.Series
        Complete training target values.

    best_params : dict
        Best hyperparameters returned by Optuna.

    Returns
    -------
    XGBRegressor
        Final trained XGBoost model.
    """

    logger.info(
        "Training optimized XGBoost model."
    )

    model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        x_train,
        y_train,
    )

    logger.info(
        "Optimized XGBoost training completed."
    )

    return model