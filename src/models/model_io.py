from __future__ import annotations

import joblib
from pathlib import Path

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