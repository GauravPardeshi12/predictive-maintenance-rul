import pandas as pd

from src.models.evaluate import calculate_metrics


def test_calculate_metrics():
    result = calculate_metrics(pd.Series([10, 20]), [10, 25])
    assert round(result["MAE"], 3) == 2.5
    assert round(result["RMSE"], 3) == 3.536
