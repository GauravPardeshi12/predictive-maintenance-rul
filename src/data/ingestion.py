from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import config
from src.utils.logger import logger


NUM_SENSORS = config["dataset"]["num_sensors"]
RAW_DATA_DIR = Path(config["paths"]["raw_data"])
BASE_COLUMNS = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]
COLUMN_NAMES = BASE_COLUMNS + SENSOR_COLUMNS


def load_single_dataset(file_path: Path) -> pd.DataFrame:
    """Load one NASA CMAPSS train or test file."""
    logger.info(f"Loading dataset {file_path.name}")

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
        usecols=COLUMN_NAMES,
    )
    df["dataset_id"] = file_path.stem.split("_")[1]

    logger.info(f"Loaded {len(df):,} rows from {file_path.name}")
    return df


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RUL from the last observed cycle of each engine."""
    logger.info("Calculating RUL.")

    df = df.copy()
    max_cycle = df.groupby(["dataset_id", "unit_id"])["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]

    logger.info(f"RUL calculated for {len(df):,} rows")
    return df


def _load_files(pattern: str) -> pd.DataFrame:
    files = sorted(RAW_DATA_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {RAW_DATA_DIR}")

    datasets = [load_single_dataset(path) for path in files]
    combined = pd.concat(datasets, ignore_index=True)

    logger.info(f"Combined {len(datasets)} datasets with final shape {combined.shape}")
    return combined


def load_training_data() -> pd.DataFrame:
    """Load all NASA CMAPSS training datasets and calculate RUL."""
    return calculate_rul(_load_files("train_*.txt"))


def load_test_data() -> pd.DataFrame:
    """Load and combine all NASA CMAPSS test datasets."""
    return _load_files("test_*.txt")


def main() -> None:
    df = load_training_data()
    logger.info(f"Final training dataset shape: {df.shape}")


if __name__ == "__main__":
    main()
