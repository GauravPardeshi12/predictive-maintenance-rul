from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import config
from src.utils.helper import save_figure


METRICS_DIR = Path(config["paths"]["metrics"])


def create_ablation_report(results_with_cycle: dict, results_without_cycle: dict) -> pd.DataFrame:
    """Compare baseline models with and without the cycle feature."""
    rows = []
    for feature_set, results in {
        "With_cycle": results_with_cycle,
        "Without_cycle": results_without_cycle,
    }.items():
        for model_name, result in results.items():
            metrics = result.get("metrics")
            if metrics is None:
                continue
            rows.append(
                {
                    "Feature_Set": feature_set,
                    "Model": model_name,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2_Score": metrics["R2_Score"],
                }
            )
    return pd.DataFrame(rows)


def save_ablation_report(ablation_df: pd.DataFrame) -> Path:
    """Save ablation results."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "ablation_study.csv"
    ablation_df.to_csv(path, index=False)
    return path


def plot_ablation_comparison(ablation_df: pd.DataFrame) -> None:
    """Plot RMSE with and without cycle."""
    pivot = ablation_df.pivot(index="Model", columns="Feature_Set", values="RMSE")
    ax = pivot.plot(kind="bar", figsize=(11, 7))
    ax.set_title("Impact of Cycle Feature on Model Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Feature Set")
    plt.tight_layout()
    save_figure("cycle_ablation_comparison", "model_comparison")
    plt.close()
