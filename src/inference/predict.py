from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.inference.preprocessing import prepare_inference_df
from src.utils.config import config
from src.utils.logger import logger


MODEL_NAME = "optimized_xgboost"


def prepare_prediction_features(
    df: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """
    Select and arrange the features required by
    the trained model.
    """

    logger.info(
        "Preparing features for prediction."
    )

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

    logger.info(
        f"Prediction feature shape: {x.shape}"
    )

    return x


def predict_rul(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate RUL predictions from raw engine data.

    The function applies feature engineering,
    loads the trained Optimized XGBoost model,
    and returns predictions with engine details.
    """

    logger.info("=" * 70)
    logger.info("STARTING RUL PREDICTION PIPELINE")
    logger.info("=" * 70)

    processed_df = prepare_inference_df(
        df
    )

    model_path = (
        Path(config["paths"]["models"])
        / f"{MODEL_NAME}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    logger.info(
        f"Loading model from {model_path}"
    )

    model = joblib.load(model_path)

    x = prepare_prediction_features(
        df=processed_df,
        model=model,
    )

    predictions = model.predict(x)

    results = processed_df[
        [
            "dataset_id",
            "unit_id",
            "cycle",
        ]
    ].copy()

    results["Predicted_RUL"] = predictions

    logger.info(
        f"Generated {len(predictions):,} predictions."
    )

    logger.info("=" * 70)
    logger.info("RUL PREDICTION COMPLETED")
    logger.info("=" * 70)

    return results


def get_latest_engine_predictions(
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the latest RUL prediction for each engine.
    """

    logger.info(
        "Extracting latest prediction for each engine."
    )

    latest_predictions = (
        predictions_df
        .sort_values(
            ["dataset_id", "unit_id", "cycle"]
        )
        .groupby(
            ["dataset_id", "unit_id"],
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    logger.info(
        f"Created predictions for "
        f"{len(latest_predictions):,} engines."
    )

    return latest_predictions


def save_predictions(
    predictions_df: pd.DataFrame,
    file_name: str = "latest_engine_predictions.csv",
) -> Path:
    """
    Save engine RUL predictions to the reports directory.
    """

    prediction_dir = (
        Path(config["paths"]["reports"])
        / "predictions"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        prediction_dir / file_name
    )

    predictions_df.to_csv(
        file_path,
        index=False,
    )

    logger.info(
        f"Predictions saved to {file_path}"
    )

    return file_path