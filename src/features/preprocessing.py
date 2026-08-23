from __future__ import annotations
import pandas as pd
from src.utils.logger import logger
from sklearn.model_selection import train_test_split

IDENTIFIER_COLUMNS = ["unit_id", "dataset_id", "engine_id", "operating_condition"]


def create_engine_identifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a globally unique engine identifier by combining dataset_id and unit_id.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    Returns
    -------
    pd.DataFrame
        Dataset with an additional engine_id column.
    """
    logger.info("Creating unique engine identifier.")

    df["engine_id"] = df["dataset_id"] + "_" + df["unit_id"].astype(str)

    logger.info(
        f"Created engine identifiers for " f"{df['engine_id'].nunique()} engines."
    )

    return df


def select_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features and target.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    X : pd.DataFrame

    y : pd.Series
    """

    logger.info("Seperating features and target.")

    x = df.drop(columns=["RUL"])

    y = df["RUL"]

    logger.info(f"Feature matrix shape: {x.shape}")
    logger.info(f"Target vector shape: {y.shape}")

    return x, y


def split_by_engine(
    x: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset by engine while preserving the
    proportion of engines from each NASA CMAPSS dataset.

    Parameters
    ----------
    x : pd.DataFrame
        Feature matrix.

    y : pd.Series
        Target vector.

    test_size : float, default=0.2
        Fraction of engines used for testing.

    random_state : int, default=42
        Random seed.

    Returns
    -------
    x_train, x_test, y_train, y_test
    """

    logger.info("Performing stratified engine wise train/test split")

    required_columns = {"dataset_id", "unit_id"}
    missing = required_columns - set(x.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    train_indices = []
    test_indices = []

    datasets = x["dataset_id"].unique()

    print("\n")
    print("=" * 70)
    print("TRAIN / TEST SPLIT SUMMARY")
    print("=" * 70)

    for dataset in datasets:
        logger.info(f"Processing dataset {dataset}")
        dataset_x = x[x["dataset_id"] == dataset]
        engine_ids = dataset_x["engine_id"].unique()
        train_engines, test_engines = train_test_split(
            engine_ids, test_size=test_size, random_state=random_state, shuffle=True
        )
        train_idx = dataset_x[dataset_x["engine_id"].isin(train_engines)].index
        test_idx = dataset_x[dataset_x["engine_id"].isin(test_engines)].index
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

        print(f"\n{dataset}")

        print("-" * 35)

        print(f"Train Engines : {len(train_engines)}")

        print(f"Test Engines  : {len(test_engines)}")

        print(f"Train Samples : {len(train_idx):,}")

        print(f"Test Samples  : {len(test_idx):,}")

    x_train = x.loc[train_indices].copy()

    x_test = x.loc[test_indices].copy()

    y_train = y.loc[train_indices].copy()

    y_test = y.loc[test_indices].copy()

    logger.info("=" * 70)

    logger.info(f"Training Engines : " f"{x_train['engine_id'].nunique()}")

    logger.info(f"Testing Engines  : " f"{x_test['engine_id'].nunique()}")

    logger.info(f"Training Samples : {len(x_train):,}")

    logger.info(f"Testing Samples  : {len(x_test):,}")

    logger.info("=" * 70)

    return (
        x_train,
        x_test,
        y_train,
        y_test,
    )


def remove_identifier_columns(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    include_cycle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove identifier columns from the training and testing
    feature matrices.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training feature matrix.

    x_test : pd.DataFrame
        Testing feature matrix.

    include_cycle : bool, default=False
        Whether to keep the cycle feature.

    Returns
    -------
    tuple
        X_train and X_test after removing unwanted columns.
    """

    logger.info("Removing identifier columns.")

    columns_to_remove = [
        column
        for column in IDENTIFIER_COLUMNS
        if column in x_train.columns
    ]

    if not include_cycle and "cycle" in x_train.columns:
        columns_to_remove.append("cycle")

    x_train = x_train.drop(
        columns=columns_to_remove
    )

    x_test = x_test.drop(
        columns=columns_to_remove
    )

    logger.info(
        f"Removed {len(columns_to_remove)} column(s): "
        f"{', '.join(columns_to_remove)}"
    )

    logger.info(
        f"Training feature shape : {x_train.shape}"
    )

    logger.info(
        f"Testing feature shape  : {x_test.shape}"
    )

    return (
        x_train,
        x_test,
    )


def prepare_training_data(
    df: pd.DataFrame,
    include_cycle: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Execute the complete preprocessing pipeline for model training.

    Parameters
    ----------
    df : pd.DataFrame
        Feature engineered dataset.

    include_cycle : bool, default=False
        Whether to keep cycle as a model feature.

    Returns
    -------
    tuple
        X_train,
        X_test,
        y_train,
        y_test,
        train_groups
    """

    logger.info("=" * 70)
    logger.info("Starting preprocessing pipeline.")
    logger.info("=" * 70)

    df = create_engine_identifier(df)

    x, y = select_features_and_target(df)

    x_train, x_test, y_train, y_test = split_by_engine(
        x=x,
        y=y,
    )

    train_groups = x_train["engine_id"].copy()

    x_train, x_test = remove_identifier_columns(
        x_train=x_train,
        x_test=x_test,
        include_cycle=include_cycle,
    )

    logger.info("=" * 70)
    logger.info("Preprocessing pipeline completed successfully.")
    logger.info("=" * 70)

    return (
        x_train,
        x_test,
        y_train,
        y_test,
        train_groups,
    )