from __future__ import annotations
import pandas as pd
from src.utils.config import config
from src.utils.logger import logger
from xgboost import XGBRegressor

def train_xgboost(x_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """
    Train an XGBoost Regressor.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series
        Training target values.

    Returns
    -------
    XGBRegressor
        Trained XGBoost model.
    """
    logger.info("Training xgboost model.")

    model = XGBRegressor(n_estimators= config["xgboost"]["n_estimators"],
                         learning_rate= config["xgboost"]["learning_rate"],
                         max_depth= config["xgboost"]["max_depth"],
                         subsample= config["xgboost"]["subsample"],
                         colsample_bytree= config["xgboost"]["colsample_bytree"],
                         random_state= config["project"]["random_seed"],
                         n_jobs= -1,
                         verbosity= config["xgboost"]["verbosity"])
    
    model.fit(x_train, y_train)

    logger.info("Training completed")

    return model

