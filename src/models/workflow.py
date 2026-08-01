from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from src.models.training.linear_regression import train_linear_regression, prepare_scaled_features
from src.models.training.random_forest import train_random_forest
from src.models.evaluate import plot_actual_vs_predicted
from src.models.evaluate import evaluate_model
from src.models.model_io import save_model, save_scaler
from src.utils.logger import logger


def _finalize_model_pipeline(
    model: RegressorMixin,
    model_name: str,
    x_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
    scaler=None,
) -> dict:
    """
    Evaluate and save a trained model.

    Parameters
    ----------
    model : RegressorMixin
        Trained model.

    model_name : str
        Model name.

    x_test : np.ndarray | pd.DataFrame
        Testing feature matrix.

    y_test : pd.Series
        Testing targets.

    scaler : object, optional
        Fitted scaler to save.

    Returns
    -------
    dict
        Model evaluation results.
    """


    logger.info(f"Finalizing {model_name} pipeline.")

    results = evaluate_model(model= model, x_test= x_test, y_test= y_test)

    plot_actual_vs_predicted(y_true= y_test, y_pred= results["predictions"], model_name= model_name)

    save_model(model= model, model_name= model_name)

    if scaler is not None:
        save_scaler(scaler= scaler, model_name= model_name)

    logger.info(f"{model_name} pipeline completed.")

    return results


def run_linear_regression(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Execute the complete Linear Regression workflow.
    """

    logger.info(
        "Running Linear Regression workflow."
    )

    x_train_scaled, x_test_scaled, scaler = (
        prepare_scaled_features(
            x_train=x_train,
            x_test=x_test,
        )
    )

    model = train_linear_regression(
        x_train=x_train_scaled,
        y_train=y_train,
    )

    return _finalize_model_pipeline(
        model=model,
        model_name="linear_regression",
        x_test=x_test_scaled,
        y_test=y_test,
        scaler=scaler,
    )


def run_random_forest(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Execute the complete Random Forest workflow.
    """

    logger.info(
        "Running Random Forest workflow."
    )

    model = train_random_forest(
        x_train=x_train,
        y_train=y_train,
    )

    return _finalize_model_pipeline(
        model=model,
        model_name="random_forest",
        x_test=x_test,
        y_test=y_test,
    )


def run_all_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    """
    Execute all baseline models.
    """

    logger.info(
        "Running baseline models."
    )

    results = {
        "Linear Regression": run_linear_regression(
            x_train,
            x_test,
            y_train,
            y_test,
        ),
        "Random Forest": run_random_forest(
            x_train,
            x_test,
            y_train,
            y_test,
        ),
    }

    logger.info(
        "Baseline models completed."
    )

    return results