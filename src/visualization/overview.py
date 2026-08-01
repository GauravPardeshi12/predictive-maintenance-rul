import pandas as pd
from src.utils.config import logger


def dataset_overview(df: pd.DataFrame) -> None:
    """
    Display basic information about dataset

    Parameters
    -----------
    df: pd.DataFrame
        Input dataset
    """

    logger.info("Generating dataset overview.")

    print("\n")
    print("=" * 60)
    print("DATA OVERVIEW")
    print("=" * 60)

    print(f"shape               : {df.shape}")
    print(
        f"No. of engines      : {df[['dataset_id','unit_id']].drop_duplicates().shape[0]}"
    )
    print(f"No. of Datasets     : {df['dataset_id'].nunique()}")
    print(f"Memory usage(mb)    : {df.memory_usage(deep=True).sum() / 1024**2:.2f}")
    engine_count = (
        df[["dataset_id", "unit_id"]].drop_duplicates().groupby("dataset_id").size()
    )

    print("-" * 60)

    print("Engine per dataset")
    print(engine_count)

    print("-" * 60)

    print("No. of rows per dataset")
    print(df["dataset_id"].value_counts().sort_values())

    print("-" * 60)

    print("\nData types")
    df.info(memory_usage="deep")

    logger.info("Dataset overview generated successfuly.")


def analyze_missing_values(df: pd.DataFrame) -> None:
    """
    analyze missing values in the dataset

    Parameters
    ----------
    df: pd.DataFrame
        Input dataset
    """

    logger.info("Analyzing missing values.")

    missing_summary = df.isnull().sum().to_frame(name="Missing Values")

    missing_summary["Percentage"] = (missing_summary["Missing Values"] / len(df)) * 100

    missing_summary = missing_summary[
        missing_summary["Missing Values"] > 0
    ].sort_values(by="Missing Values", ascending=False)

    print("\n")
    print("=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    if missing_summary.empty:
        print("No missing values found.")
    else:
        print(missing_summary)

    logger.info("Missing value analysis completed.")
