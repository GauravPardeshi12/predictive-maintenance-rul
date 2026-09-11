from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.models.evaluate import (
    calculate_residuals,
    plot_actual_vs_predicted,
    plot_residual_distribution,
    plot_residuals_vs_predictions,
)
from src.utils.config import config
from src.utils.helper import save_figure
from src.utils.logger import logger


METRICS_DIR = Path(config["paths"]["metrics"])


def load_test_predictions() -> pd.DataFrame:
    """Load the final-cycle predictions produced by unseen-test evaluation."""
    path = METRICS_DIR / "nasa_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    df = pd.read_csv(path)
    required = {"dataset_id", "unit_id", "cycle", "Predicted_RUL", "Actual_RUL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def create_residual_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add signed and absolute prediction errors."""
    df = df.copy()
    df["Residual"] = calculate_residuals(df["Actual_RUL"], df["Predicted_RUL"])
    df["Absolute_Error"] = df["Residual"].abs()
    return df


def create_dataset_error_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize error behaviour by CMAPSS dataset."""
    summary = (
        df.groupby("dataset_id")
        .agg(
            Engines=("unit_id", "count"),
            MAE=("Absolute_Error", "mean"),
            Mean_Residual=("Residual", "mean"),
            Maximum_Error=("Absolute_Error", "max"),
        )
        .reset_index()
    )

    rmse = df.groupby("dataset_id")["Residual"].apply(
        lambda values: (values.pow(2).mean()) ** 0.5
    )
    summary["RMSE"] = summary["dataset_id"].map(rmse)

    return summary[
        ["dataset_id", "Engines", "MAE", "RMSE", "Mean_Residual", "Maximum_Error"]
    ].round(3)


def save_error_summary(summary: pd.DataFrame) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "nasa_test_error_analysis.csv"
    summary.to_csv(path, index=False)
    return path


def save_worst_predictions(df: pd.DataFrame, top_n: int = 20) -> Path:
    path = METRICS_DIR / "worst_test_predictions.csv"
    df.nlargest(top_n, "Absolute_Error").to_csv(path, index=False)
    return path


def plot_error_by_dataset(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(summary["dataset_id"], summary["RMSE"])
    plt.title("Unseen Test RMSE by Dataset")
    plt.xlabel("Dataset")
    plt.ylabel("RMSE")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_figure("unseen_test_rmse_by_dataset", "error_analysis")
    plt.close()


def plot_absolute_error_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df["Absolute_Error"], bins=40, alpha=0.75)
    plt.axvline(
        df["Absolute_Error"].mean(),
        linestyle="--",
        linewidth=2,
        label="Mean Absolute Error",
    )
    plt.title("Absolute Prediction Error Distribution")
    plt.xlabel("Absolute Error")
    plt.ylabel("Number of Engines")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_figure("absolute_error_distribution", "error_analysis")
    plt.close()


def run_error_analysis() -> pd.DataFrame:
    """Run the final error analysis on unseen test predictions."""
    df = create_residual_data(load_test_predictions())
    summary = create_dataset_error_summary(df)

    save_error_summary(summary)
    save_worst_predictions(df)

    plot_actual_vs_predicted(
        df["Actual_RUL"],
        df["Predicted_RUL"].to_numpy(),
        "Optimized XGBoost - NASA Test",
    )
    plot_residual_distribution(
        df["Actual_RUL"],
        df["Predicted_RUL"].to_numpy(),
        "Optimized XGBoost - NASA Test",
    )
    plot_residuals_vs_predictions(
        df["Actual_RUL"],
        df["Predicted_RUL"].to_numpy(),
        "Optimized XGBoost - NASA Test",
    )
    plot_error_by_dataset(summary)
    plot_absolute_error_distribution(df)

    return summary


if __name__ == "__main__":
    print(run_error_analysis().to_string(index=False))
