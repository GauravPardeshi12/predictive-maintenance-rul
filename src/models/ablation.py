from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import config
from src.utils.logger import logger
from src.utils.plotting import save_figure

def create_ablation_report(
    results_with_cycle: dict,
    results_without_cycle: dict,
) -> pd.DataFrame:
    """
    Create a comparison between models trained with and without cycle.
    """

    logger.info("Creating ablation study report.")

    row = []

    experiments = {"With_cycle": results_with_cycle,
                   "Without_cycle": results_without_cycle}

    for feature_set, results in experiments.items():
        for model_name, model_results in results.items():
            if not isinstance(model_results, dict):
                continue
            metrics = model_results.get("metrics")

            if metrics is None:
                continue

            row.append({"Feature_Set": feature_set,
                        "Model": model_name,
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "R2_Score": metrics["R2_Score"]})

    ablation_df = pd.DataFrame(row) 

    logger.info("Ablation report created with"
                f"{len(ablation_df)} model results")

    return ablation_df


def save_ablation_report(ablation_df: pd.DataFrame) -> Path:
    """
    Save the ablation study results to the metrics directory.
    """

    metrics_dir = Path(config["path"]["metrics"])

    metrics_dir.mkdir(parents= True, exist_ok = True)

    report_path = metrics_dir / "ablation_study.csv"

    ablation_df.to_csv(report_path, index= False)

    logger.info(f"Ablation report saved to {report_path}")

    return report_path


def plot_ablation_comparison(
    ablation_df: pd.DataFrame,
) -> None:
    """
    Plot RMSE comparison for models trained with
    and without the cycle feature.
    """

    logger.info("Generating ablation study comparison.")

    pivot_df = ablation_df.pivot(index= "Model", columns= "Feature_Set", values= "RMSE")

    ax = pivot_df.plot(kind= "bar", figsize= (11,7))

    ax.set_title(
        "Impact of Cycle Feature on Model Performance",
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Model",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_ylabel(
        "RMSE",
        fontsize=13,
        fontweight="bold",
    )

    ax.tick_params(
        axis="x",
        labelrotation=0,
        labelsize=11,
    )

    ax.tick_params(
        axis="y",
        labelsize=11,
    )

    ax.grid(
        axis="y",
        alpha=0.3,
    )

    ax.legend(
        title="Feature Set",
    )

    plt.tight_layout()

    save_figure(
        figure_name="cycle_ablation_comparison",
        subfolder="model_comparison",
    )

    plt.show()
    plt.close()

    logger.info("Ablation study comparison generated.")