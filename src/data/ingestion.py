import pandas as pd
from pathlib import Path
from src.utils.logger import logger
from src.utils.config import config


NUM_SENSORS = config["dataset"]["num_sensors"]
RAW_DATA_DIR = Path(config["paths"]["raw_data"])

def load_single_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load single NASA CMAPSS training dataset

    Parameters
    -----------
    file_path: Path
        Path to the dataset file

    Return
    ----------
    pd.DataFrame
        Loaded Dataset with proper column names.
    """

    logger.info(f"Loading Dataset {file_path.name}")

    if not file_path.exists():
        logger.error("File Not Found")
        raise FileNotFoundError("File Not Found")

    base_columns = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    sensor_columns = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]

    column_names = base_columns + sensor_columns

    df = pd.read_csv(
        file_path, 
        sep=r"\s+", 
        header= None , 
        names=column_names, 
        usecols=column_names)
    df["dataset_id"] = file_path.stem.split("_")[1]

    logger.info(f"Successfully loaded {file_path.name}")

    return df


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Remaining Useful Life (RUL) for each engine

    Parameters
    -----------
    df: pd.DataFrame
        Input Dataset containing engine cycles

    Returns
    -----------
    df: pd.DataFrame
        Dataset with an additional RUL column
    """

    logger.info("Calculating the Remaining Useful Life (RUL).")
    max_cycles = (
        df.groupby(["dataset_id", "unit_id"])["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "max_cycle"})
    )

    df = df.merge(max_cycles, on=["dataset_id","unit_id"], how="left")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df = df.drop(columns=["max_cycle"])

    logger.info(f"RUL Calculated Successfully for the {len(df):,} Records")

    return df


def load_training_data() -> pd.DataFrame:
    """
    Load all NASA CMAPSS datasets and calculate RUL.

    Returns
    ----------
    pd.DataFrame
        combined training dataset with RUL
    """

    logger.info("Loading all training datasets.")


    dataset_files = sorted(RAW_DATA_DIR.glob("train_*.txt"))

    if not dataset_files:
        logger.error("No training dataset found")
        raise FileNotFoundError(f"No training datasets found in {RAW_DATA_DIR}")
    
    datasets = []
    for file_path in dataset_files:
        dataset = load_single_dataset(Path(file_path))
        datasets.append(dataset)

    combined_df = pd.concat(datasets, ignore_index= True )

    logger.info(f"Successfully Combined {len(datasets)} datasets.")
    combined_df = calculate_rul(combined_df)

    return combined_df

def main() -> None:
    """
    Run the data ingestion pipeline.
    """
    df = load_training_data()

    logger.info(f"Final dataset shape: {df.shape}")

if __name__ == "__main__":
    main()