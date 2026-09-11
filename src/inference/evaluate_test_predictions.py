from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.ingestion import load_test_data
from src.inference.predict import predict_rul
from src.models.evaluate import calculate_metrics
from src.utils.config import config
from src.utils.logger import logger


RAW_DATA_DIR = Path(config["paths"]["raw_data"])
METRICS_DIR = Path(config["paths"]["metrics"])


def load_ground_truth_rul() -> pd.DataFrame:
    """Load the NASA test-set RUL supplied for each engine."""
    files = sorted(RAW_DATA_DIR.glob("RUL_*.txt"))
    if not files:
        raise FileNotFoundError(f"No RUL files found in {RAW_DATA_DIR}")

    frames = []
    for path in files:
        rul = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            names=["Actual_RUL"],
            usecols=[0],
        )
        rul["dataset_id"] = path.stem.split("_")[1]
        rul["unit_id"] = np.arange(1, len(rul) + 1)
        frames.append(rul)

    return pd.concat(frames, ignore_index=True)


def get_final_cycle_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Keep the prediction made at each engine's final observed cycle."""
    required = {"dataset_id", "unit_id", "cycle", "Predicted_RUL"}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    final_rows = predictions_df.groupby(["dataset_id", "unit_id"])["cycle"].idxmax()
    return (
        predictions_df.loc[final_rows, ["dataset_id", "unit_id", "cycle", "Predicted_RUL"]]
        .sort_values(["dataset_id", "unit_id"])
        .reset_index(drop=True)
    )


def evaluate_by_dataset(evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate metrics for each dataset and for all test engines combined."""
    rows = []

    for dataset_id, data in evaluation_df.groupby("dataset_id"):
        rows.append(
            {
                "Dataset": dataset_id,
                "Engines": len(data),
                **calculate_metrics(data["Actual_RUL"], data["Predicted_RUL"].to_numpy()),
            }
        )

    rows.append(
        {
            "Dataset": "Overall",
            "Engines": len(evaluation_df),
            **calculate_metrics(
                evaluation_df["Actual_RUL"],
                evaluation_df["Predicted_RUL"].to_numpy(),
            ),
        }
    )

    return pd.DataFrame(rows).round(3)


def save_evaluation_reports(evaluation_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """Save engine-level test predictions and evaluation metrics."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(METRICS_DIR / "nasa_test_predictions.csv", index=False)
    metrics_df.to_csv(METRICS_DIR / "nasa_test_evaluation.csv", index=False)


def run_test_evaluation() -> pd.DataFrame:
    """Run evaluation on NASA's completely unseen test engines."""
    logger.info("Starting NASA unseen-test evaluation")

    predictions = predict_rul(load_test_data())
    final_predictions = get_final_cycle_predictions(predictions)
    ground_truth = load_ground_truth_rul()

    evaluation_df = final_predictions.merge(
        ground_truth,
        on=["dataset_id", "unit_id"],
        how="left",
        validate="one_to_one",
    )

    if evaluation_df["Actual_RUL"].isna().any():
        missing = int(evaluation_df["Actual_RUL"].isna().sum())
        raise ValueError(f"Missing ground-truth RUL for {missing} engines")

    metrics_df = evaluate_by_dataset(evaluation_df)
    save_evaluation_reports(evaluation_df, metrics_df)

    logger.info("NASA unseen-test evaluation completed")
    return metrics_df


if __name__ == "__main__":
    print(run_test_evaluation().to_string(index=False))
