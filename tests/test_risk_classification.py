import pandas as pd

from src.inference.risk_classification import classify_maintenance_risk


def test_risk_thresholds():
    df = pd.DataFrame({"Predicted_RUL": [10, 30, 31, 60, 61]})
    result = classify_maintenance_risk(df)
    assert result["Maintenance_Status"].tolist() == [
        "Critical", "Critical", "Warning", "Warning", "Healthy"
    ]
