from __future__ import annotations

import pandas as pd
from xgboost import XGBRegressor

from src.utils.config import config


def train_optimized_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
) -> XGBRegressor:
    """Train the final XGBoost model with a capped RUL target."""
    model = XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=config["project"]["random_seed"],
        n_jobs=-1,
    )

    rul_cap = config["training"]["rul_cap"]
    model.fit(x_train, y_train.clip(upper=rul_cap))
    return model
