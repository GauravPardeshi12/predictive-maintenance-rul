from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.utils.config import config
from src.utils.logger import logger


MODEL_DIR = Path(config["paths"]["models"])


def save_model(model, model_name: str) -> Path:
    """Save a trained model to the models directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{model_name}.joblib"
    joblib.dump(model, path)
    logger.info(f"Saved model to {path}")
    return path


def save_scaler(scaler, model_name: str) -> Path:
    """Save a fitted scaler for models that require one."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{model_name}_scaler.joblib"
    joblib.dump(scaler, path)
    logger.info(f"Saved scaler to {path}")
    return path


def save_model_parameters(parameters: dict, model_name: str) -> Path:
    """Save the selected model parameters as JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{model_name}_parameters.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(parameters, file, indent=2)
    logger.info(f"Saved model parameters to {path}")
    return path
