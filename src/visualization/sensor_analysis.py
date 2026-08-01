from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.logger import logger
from src.utils.dataset_utils import get_sensor_columns
from src.utils.helper import save_figure


def plot_sensor_trend(
    df: pd.DataFrame, dataset_id: str, unit_id: int, sensor: str
) -> None:
    """
    Plot a sensor value over engine cycles.

    Parameters
    ------------
    df: pd.DataFrame
        Input dataset

    dataset_id: str
        Dataset identifier (eg. FD001)

    unit_id: int
        Engine id

    sensor: str
        Sensor column name.
    """
    logger.info(f"Plotting {sensor} for {dataset_id} -Engine {unit_id}")

    engine_df = df[(df["dataset_id"] == dataset_id) & (df["unit_id"] == unit_id)]

    plt.figure(figsize=(10, 6))
    plt.plot(engine_df["cycle"], engine_df[sensor], linewidth=2)
    plt.title(f"{sensor} Trend")
    plt.xlabel("Cycle")
    plt.ylabel(f"{sensor}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_figure(figure_name= f"{dataset_id}_{sensor}_engine_{unit_id}",subfolder= "eda")
    plt.show()
    plt.close()

    logger.info("Sensor trend plotted successfully.")


def plot_all_sensor_trends(df: pd.DataFrame, dataset_id: str, unit_id: int) -> None:
    """
    Compare multiple sensor trends for single engine

    Parameters
    -------------
    df: DataFrame
        Input dataset

    dataset_id: str
        Dataset identifier (eg. FD001)

    unit_id: int
        Engine id

    """

    logger.info(f"Plotting all sensor trends for {dataset_id} -Engine{unit_id}")

    engine_df = df[(df["dataset_id"] == dataset_id) & (df["unit_id"] == unit_id)]

    sensor_columns = get_sensor_columns()

    fig, axes = plt.subplots(
        len(sensor_columns), 1, figsize=(12, 3 * len(sensor_columns)), sharex=True
    )

    if len(sensor_columns) == 1:
        axes = [axes]

    for ax, sensor in zip(axes, sensor_columns):
        ax.plot(engine_df["cycle"], engine_df[sensor], linewidth=1.5)
        ax.set_title(f"{sensor} vs Cycle")
        # ax.set_ylabel(sensor)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Cycle")

    fig.suptitle(f"All sensor trends {dataset_id} -Engine {unit_id}", fontsize=16)
    plt.tight_layout()
    plt.show()

    logger.info("All sensor trends plotted successfully.")
