from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from src.utils.config import config


def train_xgboost(x_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train the baseline XGBoost model."""
    params = config["xgboost"]
    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=config["project"]["random_seed"],
        n_jobs=-1,
        verbosity=params["verbosity"],
        objective="reg:squarederror",
    )
    model.fit(x_train, y_train)
    return model
