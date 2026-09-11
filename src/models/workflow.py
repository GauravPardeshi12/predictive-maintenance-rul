from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.evaluate import (
    evaluate_model,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residual_distribution,
    plot_residuals_vs_predictions,
    save_feature_importance,
)
from src.models.explainability import save_shap_report
from src.models.model_io import save_model, save_model_parameters, save_scaler
from src.models.training.linear_regression import (
    prepare_scaled_features,
    train_linear_regression,
)
from src.models.training.random_forest import train_random_forest
from src.models.training.xgboost import train_xgboost
from src.models.training.xgboost_optimized import train_optimized_xgboost
from src.models.tuning.xgboost_optuna import optimize_xgboost
from src.utils.logger import logger


def _finalize_model(
    model,
    model_name: str,
    x_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
    scaler=None,
) -> dict:
    """Evaluate, plot and save a trained model."""
    results = evaluate_model(model, x_test, y_test)

    plot_actual_vs_predicted(y_test, results["predictions"], model_name)
    plot_residual_distribution(y_test, results["predictions"], model_name)
    plot_residuals_vs_predictions(y_test, results["predictions"], model_name)

    save_model(model, model_name)
    if scaler is not None:
        save_scaler(scaler, model_name)

    logger.info(
        f"{model_name}: MAE={results['metrics']['MAE']:.3f}, "
        f"RMSE={results['metrics']['RMSE']:.3f}, "
        f"R2={results['metrics']['R2_Score']:.3f}"
    )
    return results


def run_linear_regression(x_train, x_test, y_train, y_test) -> dict:
    x_train_scaled, x_test_scaled, scaler = prepare_scaled_features(x_train, x_test)
    model = train_linear_regression(x_train_scaled, y_train)
    return _finalize_model(
        model,
        "linear_regression",
        x_test_scaled,
        y_test,
        scaler,
    )


def run_random_forest(x_train, x_test, y_train, y_test) -> dict:
    model = train_random_forest(x_train, y_train)
    results = _finalize_model(model, "random_forest", x_test, y_test)

    importance = plot_feature_importance(
        model=model,
        feature_names=x_train.columns,
        model_name="random_forest",
    )
    save_feature_importance(importance, "random_forest")
    results["feature_importance"] = importance
    return results


def run_xgboost(x_train, x_test, y_train, y_test) -> dict:
    model = train_xgboost(x_train, y_train)
    return _finalize_model(model, "xgboost", x_test, y_test)


def run_optimized_xgboost(
    x_train,
    x_test,
    y_train,
    y_test,
    train_groups,
) -> dict:
    tuning_results = optimize_xgboost(
        x_train=x_train,
        y_train=y_train,
        groups=train_groups,
    )

    best_params = tuning_results["best_params"]
    save_model_parameters(best_params, "optimized_xgboost")

    model = train_optimized_xgboost(
        x_train=x_train,
        y_train=y_train,
        best_params=best_params,
    )

    importance = plot_feature_importance(
        model=model,
        feature_names=x_train.columns,
        model_name="optimized_xgboost",
    )
    save_feature_importance(importance, "optimized_xgboost")

    results = _finalize_model(
        model,
        "optimized_xgboost",
        x_test,
        y_test,
    )

    shap_sample = x_test.sample(min(5000, len(x_test)), random_state=42)
    save_shap_report(model, shap_sample, "optimized_xgboost")

    results["best_params"] = best_params
    results["best_rmse"] = tuning_results["best_rmse"]
    return results


def run_all_models(
    x_train,
    x_test,
    y_train,
    y_test,
    train_groups,
    models_to_run: list[str] | None = None,
) -> dict:
    """Run the requested model workflows."""
    models_to_run = models_to_run or [
        "linear_regression",
        "random_forest",
        "xgboost",
    ]

    runners = {
        "linear_regression": run_linear_regression,
        "random_forest": run_random_forest,
        "xgboost": run_xgboost,
    }

    results = {}
    for model_name in models_to_run:
        if model_name == "optimized_xgboost":
            results["Optimized XGBoost"] = run_optimized_xgboost(
                x_train, x_test, y_train, y_test, train_groups
            )
            continue

        if model_name not in runners:
            raise ValueError(f"Unknown model: {model_name}")

        display_name = {
            "linear_regression": "Linear Regression",
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
        }[model_name]
        results[display_name] = runners[model_name](
            x_train, x_test, y_train, y_test
        )

    return results
