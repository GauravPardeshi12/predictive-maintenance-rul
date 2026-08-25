from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.utils.config import config
from src.utils.helper import save_figure
from src.utils.logger import logger


def calculate_shap_values(
    model,
    x_data: pd.DataFrame,
):
    """
    Calculate SHAP values for a trained tree-based model.
    """

    logger.info("Calculating SHAP values.")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(x_data)

    logger.info("SHAP values calculated successfully.")

    return shap_values


def plot_shap_summary(
    model,
    x_data: pd.DataFrame,
    max_display: int = 20,
) -> None:
    """
    Generate a SHAP summary plot showing
    global feature impact.
    """

    logger.info("Generating SHAP summary plot.")

    shap_values = calculate_shap_values(
        model=model,
        x_data=x_data,
    )

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        x_data,
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    save_figure(
        figure_name="optimized_xgboost_shap_summary",
        subfolder="shap",
    )

    plt.close()

    logger.info(
        "SHAP summary plot saved successfully."
    )


def plot_shap_bar(
    model,
    x_data: pd.DataFrame,
    max_display: int = 20,
) -> None:
    """
    Generate a SHAP bar plot showing
    global feature importance.
    """

    logger.info("Generating SHAP feature importance plot.")

    shap_values = calculate_shap_values(
        model=model,
        x_data=x_data,
    )

    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values,
        x_data,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )

    plt.tight_layout()

    save_figure(
        figure_name="optimized_xgboost_shap_importance",
        subfolder="shap",
    )

    plt.close()

    logger.info(
        "SHAP feature importance plot saved successfully."
    )


def save_shap_importance(
    model,
    x_data: pd.DataFrame,
) -> Path:
    """
    Calculate and save mean absolute SHAP importance.
    """

    logger.info("Saving SHAP feature importance.")

    shap_values = calculate_shap_values(
        model=model,
        x_data=x_data,
    )

    importance_df = pd.DataFrame(
        {
            "Feature": x_data.columns,
            "Mean_Absolute_SHAP": abs(shap_values).mean(axis=0),
        }
    )

    importance_df = importance_df.sort_values(
        by="Mean_Absolute_SHAP",
        ascending=False,
    ).reset_index(drop=True)

    metrics_dir = Path(
        config["paths"]["metrics"]
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        metrics_dir /
        "optimized_xgboost_shap_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        f"SHAP importance saved to {output_path}."
    )

    return output_path