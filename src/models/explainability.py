from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.utils.config import config
from src.utils.helper import save_figure
from src.utils.logger import logger


def calculate_shap_values(model, x_data: pd.DataFrame):
    """Calculate SHAP values for the supplied rows."""
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(x_data)


def save_shap_report(
    model,
    x_data: pd.DataFrame,
    model_name: str = "optimized_xgboost",
    max_display: int = 20,
) -> Path:
    """Create the SHAP plots and save mean absolute SHAP importance."""
    logger.info("Generating SHAP report")

    shap_values = calculate_shap_values(model, x_data)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        x_data,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    save_figure(f"{model_name}_shap_summary", "shap")
    plt.close()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        x_data,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    save_figure(f"{model_name}_shap_importance", "shap")
    plt.close()

    importance_df = (
        pd.DataFrame(
            {
                "Feature": x_data.columns,
                "Mean_Absolute_SHAP": abs(shap_values).mean(axis=0),
            }
        )
        .sort_values("Mean_Absolute_SHAP", ascending=False)
        .reset_index(drop=True)
    )

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output_path = metrics_dir / f"{model_name}_shap_importance.csv"
    importance_df.to_csv(output_path, index=False)

    logger.info(f"SHAP report saved to {output_path}")
    return output_path
