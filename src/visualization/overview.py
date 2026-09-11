from __future__ import annotations

import pandas as pd

from src.utils.logger import logger


def dataset_overview(df: pd.DataFrame) -> None:
    """Print a compact overview of the training data."""
    engine_count = df[["dataset_id", "unit_id"]].drop_duplicates().groupby("dataset_id").size()

    print("\n" + "=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)
    print(f"Shape              : {df.shape}")
    print(f"Engines            : {df[['dataset_id', 'unit_id']].drop_duplicates().shape[0]}")
    print(f"Datasets           : {df['dataset_id'].nunique()}")
    print(f"Memory usage (MB)  : {df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    print("\nEngines per dataset")
    print(engine_count)

    logger.info("Dataset overview generated")


def analyze_missing_values(df: pd.DataFrame) -> None:
    """Print columns containing missing values."""
    missing = df.isna().sum().to_frame("Missing Values")
    missing["Percentage"] = missing["Missing Values"] / len(df) * 100
    missing = missing[missing["Missing Values"] > 0].sort_values("Missing Values", ascending=False)

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)
    print("No missing values found." if missing.empty else missing)
