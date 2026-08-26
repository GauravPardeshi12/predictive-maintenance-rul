from __future__ import annotations

import joblib
from pathlib import Path
import json


from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler

from src.utils.config import config
from src.utils.logger import logger


def save_model(model: RegressorMixin, model_name: str) -> Path:
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : RegressorMixin
        Trained model.

    model_name : str
        Name of the model.

    Returns
    -------
    Path
        Saved model path.
    """

    logger.info(f"Saving {model_name} model.")

    model_dir = Path(config["paths"]["models"])

    model_dir.mkdir(parents= True, exist_ok= True)

    model_path = model_dir / f"{model_name}.joblib"

    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")

    return model_path


def load_model(model_name: str) -> RegressorMixin:
    """
    Load a trained model from disk.

    Parameters
    ----------
    model_name : str
        Name of the saved model.

    Returns
    -------
    RegressorMixin
        Loaded trained model.
    """

    logger.info(f"Loading {model_name} model.")

    model_dir = Path(config["paths"]["models"])

    model_path = model_dir / f"{model_name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(model_path)

    logger.info(
        f"Model loaded successfully from {model_path}"
    )

    return model


def save_scaler(
    scaler: StandardScaler,
    model_name: str,
) -> Path:
    """
    Save a fitted scaler to disk.

    Parameters
    ----------
    scaler : StandardScaler
        Fitted StandardScaler instance.

    model_name : str
        Name of the associated model.

    Returns
    -------
    Path
        Path to the saved scaler.
    """

    logger.info(f"Saving scaler for {model_name}.")

    model_dir = Path(config["paths"]["models"])

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler_path = (
        model_dir
        / f"{model_name}_scaler.joblib"
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    logger.info(
        f"Scaler saved to {scaler_path}"
    )

    return scaler_path


def save_model_parameters(
    parameters: dict,
    model_name: str,
) -> Path:
    logger.info(
        f"Saving parameters for {model_name}."
    )

    model_dir = Path(
        config["paths"]["models"]
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_path = (
        model_dir /
        f"{model_name}_parameters.json"
    )

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            parameters,
            file,
            indent=4,
        )

    logger.info(
        f"Model parameters saved to {file_path}"
    )

    return file_path