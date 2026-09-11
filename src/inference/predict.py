from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.feature_engineering import prepare_features
from src.utils.config import config
from src.utils.logger import logger


MODEL_NAME = "optimized_xgboost"
MODEL_PATH = Path(config["paths"]["models"]) / f"{MODEL_NAME}.joblib"


def prepare_prediction_features(df: pd.DataFrame, model) -> pd.DataFrame:
    """Select the exact feature columns used when the model was trained."""
    if not hasattr(model, "feature_names_in_"):
        raise AttributeError("The trained model does not contain feature names")

    feature_names = list(model.feature_names_in_)
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df[feature_names].copy()


def predict_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Generate RUL predictions from raw CMAPSS engine data."""
    logger.info("Starting RUL prediction pipeline")

    processed_df = prepare_features(df, remove_constants=False)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    x = prepare_prediction_features(processed_df, model)
    predictions = model.predict(x)

    results = processed_df[["dataset_id", "unit_id", "cycle"]].copy()
    results["Predicted_RUL"] = predictions

    logger.info(f"Generated {len(results):,} RUL predictions")
    return results


def get_latest_engine_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest observed cycle for every engine."""
    return (
        predictions_df.sort_values(["dataset_id", "unit_id", "cycle"])
        .groupby(["dataset_id", "unit_id"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def save_predictions(
    predictions_df: pd.DataFrame,
    file_name: str = "latest_engine_predictions.csv",
) -> Path:
    """Save predictions to the reports directory."""
    prediction_dir = Path(config["paths"]["predictions"])
    prediction_dir.mkdir(parents=True, exist_ok=True)
    path = prediction_dir / file_name
    predictions_df.to_csv(path, index=False)
    logger.info(f"Predictions saved to {path}")
    return path
