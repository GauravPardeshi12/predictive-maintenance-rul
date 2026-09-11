from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import logger


IDENTIFIER_COLUMNS = ["unit_id", "dataset_id", "engine_id", "operating_condition"]


def create_engine_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unique engine identifier across all CMAPSS datasets."""
    df = df.copy()
    df["engine_id"] = df["dataset_id"] + "_" + df["unit_id"].astype(str)
    logger.info(f"Created identifiers for {df['engine_id'].nunique()} engines")
    return df


def select_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the RUL target."""
    if "RUL" not in df.columns:
        raise ValueError("RUL column is required for training")

    x = df.drop(columns="RUL")
    y = df["RUL"]
    return x, y


def split_by_engine(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split engines within each dataset so rows from one engine never cross sets."""
    required_columns = {"dataset_id", "engine_id"}
    missing = required_columns - set(x.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_indices: list[int] = []
    test_indices: list[int] = []

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT SUMMARY")
    print("=" * 70)

    for dataset_id in sorted(x["dataset_id"].unique()):
        dataset_x = x[x["dataset_id"] == dataset_id]
        engine_ids = dataset_x["engine_id"].drop_duplicates().to_numpy()

        train_engines, test_engines = train_test_split(
            engine_ids,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        train_idx = dataset_x.index[dataset_x["engine_id"].isin(train_engines)]
        test_idx = dataset_x.index[dataset_x["engine_id"].isin(test_engines)]

        train_indices.extend(train_idx.tolist())
        test_indices.extend(test_idx.tolist())

        print(f"\n{dataset_id}")
        print(f"Train Engines : {len(train_engines)}")
        print(f"Test Engines  : {len(test_engines)}")
        print(f"Train Samples : {len(train_idx):,}")
        print(f"Test Samples  : {len(test_idx):,}")

    x_train = x.loc[train_indices].copy()
    x_test = x.loc[test_indices].copy()
    y_train = y.loc[train_indices].copy()
    y_test = y.loc[test_indices].copy()

    logger.info(
        f"Split complete: {x_train['engine_id'].nunique()} train engines, "
        f"{x_test['engine_id'].nunique()} test engines"
    )
    return x_train, x_test, y_train, y_test


def remove_identifier_columns(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    include_cycle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove identifiers and optionally remove cycle for the ablation experiment."""
    columns_to_remove = [
        column for column in IDENTIFIER_COLUMNS if column in x_train.columns
    ]

    if not include_cycle and "cycle" in x_train.columns:
        columns_to_remove.append("cycle")

    x_train = x_train.drop(columns=columns_to_remove)
    x_test = x_test.drop(columns=columns_to_remove)

    logger.info(f"Removed columns: {', '.join(columns_to_remove)}")
    return x_train, x_test


def prepare_training_data(
    df: pd.DataFrame,
    include_cycle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Run engine grouping, train/test splitting and final feature selection."""
    df = create_engine_identifier(df)
    x, y = select_features_and_target(df)

    x_train, x_test, y_train, y_test = split_by_engine(
        x=x,
        y=y,
    )

    train_groups = x_train["engine_id"].copy()
    x_train, x_test = remove_identifier_columns(
        x_train,
        x_test,
        include_cycle=include_cycle,
    )

    logger.info(
        f"Training features: {x_train.shape}; test features: {x_test.shape}"
    )
    return x_train, x_test, y_train, y_test, train_groups
