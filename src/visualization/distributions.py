from __future__ import annotations
import pandas as pd
from src.utils.logger import logger
import matplotlib.pyplot as plt
from src.utils.helper import save_figure


def engine_lifetime_analysis(df: pd.DataFrame) -> None:
    """
    Analyze the lifetime distribution of engines

    Parameters
    df: pd.DataFrame
        Input dataset
    """

    logger.info("Analyzing engine lifetime.")

    engine_lifetime = df.groupby(["dataset_id", "unit_id"])["cycle"].max()

    print("\n")
    print("=" * 60)
    print("ENGINE LIFETIME SUMMARY")
    print("=" * 60)

    print(engine_lifetime.describe())

    plt.figure(figsize=(10, 6))
    plt.hist(engine_lifetime, bins=30)
    plt.title("Engine lifetime distribution")
    plt.xlabel("Lifetime (cycles)")
    plt.ylabel("Number of engines")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_figure(figure_name= "engine_lifetime_distribution", subfolder= "eda")
    plt.show()
    plt.close()

    logger.info("Engine lifetime analysis completed.")
