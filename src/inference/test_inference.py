from __future__ import annotations

from src.data.ingestion import load_test_data
from src.inference.predict import (
    get_latest_engine_predictions,
    predict_rul,
    save_predictions,
)
from src.inference.risk_classification import classify_maintenance_risk
from src.utils.logger import logger


def run_inference_test():
    """Run the trained model on the unseen CMAPSS test datasets."""
    raw_df = load_test_data()
    predictions = predict_rul(raw_df)
    latest = get_latest_engine_predictions(predictions)
    results = classify_maintenance_risk(latest)
    save_path = save_predictions(results)

    print("\n" + "=" * 70)
    print("LATEST ENGINE RUL PREDICTIONS")
    print("=" * 70)
    print(f"Total engines        : {len(results):,}")
    print(f"Mean predicted RUL   : {results['Predicted_RUL'].mean():.2f}")
    print(f"Minimum predicted RUL: {results['Predicted_RUL'].min():.2f}")
    print(f"Maximum predicted RUL: {results['Predicted_RUL'].max():.2f}")
    print("\nMaintenance Status")
    print(results["Maintenance_Status"].value_counts().to_string())
    print(f"\nPredictions saved to: {save_path}")

    logger.info("Inference test completed")
    return results


if __name__ == "__main__":
    run_inference_test()
