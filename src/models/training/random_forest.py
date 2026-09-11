from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils.config import config


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestRegressor:
    """Train the Random Forest baseline."""
    params = config["random_forest"]
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
        n_jobs=params["n_jobs"],
    )
    model.fit(x_train, y_train)
    return model
