import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils.config import config
from src.utils.logger import logger


def train_random_forest(
    x_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestRegressor:
    """
    Train a Random Forest regression model.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series
        Training target values.

    Returns
    -------
    RandomForestRegressor
        Trained Random Forest model.
    """

    logger.info("Training Random Forest Model.")

    model = RandomForestRegressor(
        n_estimators=config["random_forest"]["n_estimators"],
        max_depth=config["random_forest"]["max_depth"],
        random_state=config["random_forest"]["random_state"],
        n_jobs=config["random_forest"]["n_jobs"]
    )

    model.fit(x_train, y_train)

    logger.info("Random Forest training  completed.")

    return model

