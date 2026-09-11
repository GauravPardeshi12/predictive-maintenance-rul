from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.utils.config import config
from src.utils.helper import save_figure
from src.utils.logger import logger


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate the core regression metrics used throughout the project."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2_Score": r2_score(y_true, y_pred),
    }


def evaluate_model(
    model: RegressorMixin,
    x_test: np.ndarray | pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Generate predictions and metrics for a trained model."""
    predictions = model.predict(x_test)
    return {
        "metrics": calculate_metrics(y_test, predictions),
        "predictions": predictions,
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Build a model comparison table sorted by RMSE."""
    rows = []
    for model_name, result in results.items():
        metrics = result.get("metrics")
        if metrics is None:
            continue
        rows.append(
            {
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2_Score": metrics["R2_Score"],
            }
        )

    if not rows:
        raise ValueError("No model metrics were provided for comparison")

    return (
        pd.DataFrame(rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
        .round(3)
    )


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """Print the final model comparison."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    best = comparison_df.iloc[0]
    print(f"\nBest Model : {best['Model']}")
    print(f"RMSE      : {best['RMSE']:.3f}")


def save_model_comparison(comparison_df: pd.DataFrame) -> Path:
    """Save model comparison metrics."""
    report_dir = Path(config["paths"]["metrics"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "model_comparison.csv"
    comparison_df.to_csv(report_path, index=False)
    logger.info(f"Model comparison saved to {report_path}")
    return report_path


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Plot actual versus predicted RUL."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=12)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=2,
        color="red",
        label="Perfect Prediction",
    )

    plt.title(f"Actual vs Predicted RUL\n{model_name}")
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_figure(
        figure_name=model_name.lower().replace(" ", "_"),
        subfolder="actual_vs_predicted",
    )
    plt.close()


def plot_feature_importance(
    model,
    feature_names,
    model_name: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Plot and return split-based feature importance."""
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{model_name} does not provide feature_importances_")

    importance_df = (
        pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    top_features = importance_df.head(top_n).sort_values("Importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title(f"Top {top_n} Feature Importance\n{model_name}")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    save_figure(
        figure_name=f"{model_name}_feature_importance",
        subfolder="feature_importance",
    )
    plt.close()
    return importance_df


def save_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
) -> Path:
    """Save feature importance values."""
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path = metrics_dir / f"{model_name}_feature_importance.csv"
    importance_df.to_csv(report_path, index=False)
    return report_path


def calculate_residuals(y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
    """Calculate residuals as actual RUL minus predicted RUL."""
    return np.asarray(y_true) - np.asarray(y_pred)


def plot_residual_distribution(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Plot the distribution of residuals."""
    residuals = calculate_residuals(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=40, alpha=0.7)
    plt.axvline(0, linestyle="--", linewidth=2, color="red", label="Zero Error")
    plt.title(f"Residual Distribution\n{model_name}")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_figure(
        figure_name=f"{model_name}_residual_distribution",
        subfolder="residuals",
    )
    plt.close()


def plot_residuals_vs_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Plot residuals against predicted RUL."""
    residuals = calculate_residuals(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.4, s=12)
    plt.axhline(0, linestyle="--", linewidth=2, color="red", label="Zero Error")
    plt.title(f"Residuals vs Predicted RUL\n{model_name}")
    plt.xlabel("Predicted RUL")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_figure(
        figure_name=f"{model_name}_residuals_vs_predictions",
        subfolder="residuals",
    )
    plt.close()
