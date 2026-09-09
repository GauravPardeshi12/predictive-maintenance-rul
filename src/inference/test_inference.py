from __future__ import annotations

import pandas as pd

from src.data.ingestion import load_test_data

from src.inference.predict import (
    predict_rul,
    get_latest_engine_predictions,
    save_predictions,
)

from src.utils.logger import logger


def run_inference_test() -> pd.DataFrame:
    """
    Run the trained model on unseen NASA CMAPSS test data.
    """

    logger.info("=" * 70)
    logger.info("STARTING INFERENCE TEST")
    logger.info("=" * 70)

    raw_df = load_test_data()

    all_predictions = predict_rul(
        raw_df
    )

    latest_predictions = (
        get_latest_engine_predictions(
            all_predictions
        )
    )

    save_path = save_predictions(
        latest_predictions
    )

    print("\n")
    print("=" * 70)
    print("LATEST ENGINE RUL PREDICTIONS")
    print("=" * 70)

    print(
        f"\nTotal engines: "
        f"{len(latest_predictions):,}"
    )

    print(
        f"Mean predicted RUL: "
        f"{latest_predictions['Predicted_RUL'].mean():.2f}"
    )

    print(
        f"Minimum predicted RUL: "
        f"{latest_predictions['Predicted_RUL'].min():.2f}"
    )

    print(
        f"Maximum predicted RUL: "
        f"{latest_predictions['Predicted_RUL'].max():.2f}"
    )

    print("\nSample Predictions")
    print("-" * 70)

    print(
        latest_predictions
        .head(10)
        .to_string(index=False)
    )

    print("\nPredictions saved to:")
    print(save_path)

    logger.info("=" * 70)
    logger.info(
        "INFERENCE TEST COMPLETED SUCCESSFULLY"
    )
    logger.info("=" * 70)

    return latest_predictions


if __name__ == "__main__":
    run_inference_test()