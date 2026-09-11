from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def prepare_scaled_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a scaler on training data and transform train/test features."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_test_scaled, scaler


def train_linear_regression(
    x_train: np.ndarray,
    y_train: pd.Series,
) -> LinearRegression:
    """Train the linear-regression baseline."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model
