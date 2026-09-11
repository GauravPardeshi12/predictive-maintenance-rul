from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import config
from src.utils.logger import logger


CRITICAL_MAX_RUL = config["risk"]["critical_max_rul"]
WARNING_MAX_RUL = config["risk"]["warning_max_rul"]


def classify_maintenance_risk(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a maintenance status from predicted RUL."""
    if "Predicted_RUL" not in predictions_df.columns:
        raise ValueError("Predicted_RUL column is required")

    df = predictions_df.copy()
    df["Maintenance_Status"] = np.select(
        [
            df["Predicted_RUL"] <= CRITICAL_MAX_RUL,
            df["Predicted_RUL"] <= WARNING_MAX_RUL,
        ],
        ["Critical", "Warning"],
        default="Healthy",
    )
    return df
