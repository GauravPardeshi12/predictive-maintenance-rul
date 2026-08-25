from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.base import RegressorMixin
from src.models.training.linear_regression import (
    train_linear_regression,
    prepare_scaled_features,
)
from src.models.training.random_forest import train_random_forest
from src.models.training.xgboost import train_xgboost
from src.models.training.xgboost_optimized import train_optimized_xgboost
from src.models.tuning.xgboost_optuna import optimize_xgboost

from src.models.evaluate import plot_actual_vs_predicted, plot_feature_importance, save_feature_importance
from src.models.evaluate import evaluate_model, plot_residual_distribution, plot_residuals_vs_predictions
from src.models.model_io import save_model, save_scaler
from src.utils.logger import logger
from src.models.explainability import (
    plot_shap_summary,
    plot_shap_bar,
    save_shap_importance,
)


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

    results = evaluate_model(model=model, x_test=x_test, y_test=y_test)

    plot_actual_vs_predicted(
        y_true=y_test, y_pred=results["predictions"], model_name=model_name
    )

    plot_residual_distribution(
    y_true=y_test,
    y_pred= results["predictions"],
    model_name=model_name,
)

    plot_residuals_vs_predictions(
    y_true=y_test,
    y_pred= results["predictions"],
    model_name=model_name,
)

    save_model(model=model, model_name=model_name)

    if scaler is not None:
        save_scaler(scaler=scaler, model_name=model_name)

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

    logger.info("Running Linear Regression workflow.")

    x_train_scaled, x_test_scaled, scaler = prepare_scaled_features(
        x_train=x_train,
        x_test=x_test,
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

    logger.info("Running Random Forest workflow.")

    model = train_random_forest(
        x_train=x_train,
        y_train=y_train,
    )

    results = _finalize_model_pipeline(
        model=model,
        model_name="random_forest",
        x_test=x_test,
        y_test=y_test,
    )

    importance_df = plot_feature_importance(
        model=model, feature_names=x_train.columns, model_name="random_forest", top_n=20
    )

    save_feature_importance(importance_df= importance_df, model_name= "random_forest")

    results["feature_importance"] = importance_df
    return results


def run_xgboost(
    x_train: pd.DataFrame, x_test: pd.Series, y_train: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Execute the complete xgboost Regressor workflow.
    """

    logger.info("Running xgboost Regressor workflow")

    model = train_xgboost(x_train=x_train, y_train=y_train)

    return _finalize_model_pipeline(
        model=model, model_name="xgboost", x_test=x_test, y_test=y_test
    )

def run_optimized_xgboost(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_groups: pd.Series,
) -> dict:

    logger.info("Running optimized XGBoost workflow.")

    tuning_results = optimize_xgboost(x_train= x_train, y_train= y_train, groups= train_groups, n_trials= 30)
    best_params = tuning_results["best_params"]
    logger.info(f"Best XGBoost parameter {best_params}")

    model = train_optimized_xgboost(x_train= x_train, y_train= y_train, best_params= best_params)

    importance_df = plot_feature_importance(
    model=model,
    feature_names=x_train.columns,
    model_name="optimized_xgboost",
    top_n=20,
)
    save_feature_importance(
    importance_df=importance_df,
    model_name="optimized_xgboost",
)

    results = _finalize_model_pipeline(model= model, model_name= "optimized_xgboost", x_test= x_test, y_test= y_test)

    shap_data = x_test.sample(
    n=min(5000, len(x_test)),
    random_state=42,
    )
    plot_shap_summary(
        model=model,
        x_data=shap_data,
        max_display=20,
    )

    plot_shap_bar(
        model=model,
        x_data=shap_data,
        max_display=20,
    )

    save_shap_importance(
        model=model,
        x_data=shap_data,
    )

    results["best_params"] = best_params
    results["best_rmse"] = tuning_results["best_rmse"]

    return results


def run_all_models(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_groups: pd.Series,
    models_to_run: list[str] | None = None,
) -> dict:

    if models_to_run is None:
        models_to_run = [
            "linear_regression",
            "random_forest",
            "xgboost",
        ]

    results = {}

    if "linear_regression" in models_to_run:
        results["Linear Regression"] = run_linear_regression(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )

    if "random_forest" in models_to_run:
        results["Random Forest"] = run_random_forest(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )

    if "xgboost" in models_to_run:
        results["XGBoost"] = run_xgboost(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )

    if "optimized_xgboost" in models_to_run:
        results["Optimized XGBoost"] = run_optimized_xgboost(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            train_groups=train_groups,
        )

    return results