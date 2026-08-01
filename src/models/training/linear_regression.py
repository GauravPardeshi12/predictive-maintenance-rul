import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.utils.logger import logger

# =====================================================================================
# Data Preparation
# =====================================================================================


def prepare_scaled_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scale the training and testing data for
    Linear Regression.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.

    X_test : pd.DataFrame
        Testing feature matrix.

    Returns
    -------
    tuple
        X_train_scaled,
        X_test_scaled,
        fitted scaler.
    """

    logger.info("Scaling features for linear regression.")

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    logger.info("Feature scaling completed.")

    return (x_train_scaled, x_test_scaled, scaler)


# =====================================================================================
# Baseline models
# =====================================================================================


def train_linear_regression(
    x_train: np.ndarray, y_train: pd.Series
) -> LinearRegression:
    """
    Train a Linear Regression model.

    Parameters
    ----------
    X_train : np.ndarray
        Scaled training feature matrix.

    y_train : pd.Series
        Training target values.

    Returns
    -------
    LinearRegression
        Trained Linear Regression model.
    """

    logger.info("Training Linear Regression model.")

    model = LinearRegression()
    model.fit(x_train, y_train)

    logger.info("Linear Regression training completed.")

    return model
