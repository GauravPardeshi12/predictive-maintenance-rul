from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.inference.preprocessing import prepare_inference_data
from src.utils.config import config
from src.utils.logger import logger


MODEL_NAME = "optimized_xgboost"


def load_prediction_model():
    """
    Load the trained Optimized XGBoost model.
    """

    logger.info(
        "Loading Optimized XGBoost model."
    )

    model_dir = Path(
        config["paths"]["models"]
    )

    model_path = (
        model_dir
        / f"{MODEL_NAME}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    logger.info(
        f"Model loaded successfully from {model_path}"
    )

    return model


def prepare_prediction_features(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Prepare model input using the feature names
    stored by the trained model.
    """

    if not hasattr(model, "feature_names_in_"):
        raise AttributeError(
            "The trained model does not contain "
            "feature names."
        )

    feature_names = list(
        model.feature_names_in_
    )

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required features: "
            f"{missing_features}"
        )

    x = df[feature_names].copy()

    if not all(
        pd.api.types.is_numeric_dtype(dtype)
        for dtype in x.dtypes
    ):
        raise TypeError(
            "All prediction features must be numeric."
        )

    logger.info(
        f"Prediction feature shape: {x.shape}"
    )

    return x


def predict_rul(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Generate RUL predictions from raw engine data.
    """

    logger.info(
        "Starting RUL prediction pipeline."
    )

    df = prepare_inference_data(df)

    model = load_prediction_model()

    x = prepare_prediction_features(
        df=df,
        model=model,
    )

    predictions = model.predict(x)

    logger.info(
        f"Generated {len(predictions):,} RUL predictions."
    )

    return predictions