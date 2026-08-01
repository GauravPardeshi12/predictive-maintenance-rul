from __future__ import annotations
import pandas as pd
import numpy as np
from src.utils.logger import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.base import RegressorMixin
from src.utils.config import config
from pathlib import Path
import matplotlib.pyplot as plt
from src.utils.helper import save_figure


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """
    Calculate regression evaluation metrics.

    Parameters
    ----------
    y_true : pd.Series
        Actual target values.

    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    dict
        Dictionary containing regression metrics.
    """
    logger.info("Calculating Regression metrics.")

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2_Score": r2_score(y_true, y_pred),
    }

    logger.info("Regression metrics completed successfully.")

    return metrics


def evaluate_model(
    model: RegressorMixin, x_test: np.ndarray | pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Evaluate a regression model.

    Parameters
    ----------
    model : RegressorMixin
        Trained regression model.

    X_test : np.ndarray | pd.DataFrame
        Testing features.

    y_test : pd.Series
        Testing targets.

    Returns
    -------
    dict
        Regression metrics.
    """

    logger.info("Evaluating model")

    y_pred = model.predict(x_test)

    metrics = calculate_metrics(y_true=y_test, y_pred=y_pred)

    logger.info("Model evaluation completed.")

    return {"metrics": metrics, "predictions": y_pred}


def compare_models(results: dict) -> pd.DataFrame:
    """
    Compare the performance of all trained models.

    Parameters
    ----------
    results : dict
        Dictionary containing model evaluation results.

    Returns
    -------
    pd.DataFrame
        Model comparison table sorted by RMSE.
    """

    logger.info("Comparing model performance.")

    comparison = []

    for model_name, result in results.items():
        metrics = result["metrics"]
        comparison.append(
            {
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2_Score": metrics["R2_Score"],
            }
        )

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values(by="RMSE", ascending=True).reset_index(
        drop=True
    )
    comparison_df[["Model", "MAE", "RMSE", "R2_Score"]] = comparison_df[
        ["Model", "MAE", "RMSE", "R2_Score"]
    ].round(3)

    logger.info("Model comparison completed.")

    return comparison_df


def print_model_comparison(comparison_df: pd.DataFrame) -> None:
    """
    Display model comparison summary
    """

    logger.info("Printing model comparison.")
    print("\n")
    print("="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print()

    print(comparison_df.to_string(index= False))

    best_model = comparison_df.iloc[0]

    print(f"🏆 Best Model : {best_model['Model']}")

    print(f"RMSE {best_model['RMSE']:.3f}")

    logger.info("Model comparison printed.")


def save_model_comparison(comparison_df: pd.DataFrame) -> Path:
    """
    Save the model comparison table to the reports directory.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Model comparison table.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """

    logger.info("Saving model comparison report.")

    report_dir = Path(config["paths"]["metrics"])

    report_dir.mkdir(parents= True, exist_ok= True)

    report_path = report_dir / "model_comparison.csv"

    comparison_df.to_csv(report_path, index= False)

    logger.info(f"Model comparison report saved to {report_path}")

    return report_path


def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    """
    Plot actual versus predicted RUL values.

    Parameters
    ----------
    y_true : pd.Series
        Actual target values.

    y_pred : np.ndarray
        Predicted target values.

    model_name : str
        Name of the regression model.
    """

    logger.info(f"Generating Actual vs Predicted plot for {model_name}.")

    plt.figure(figsize= (8,8))
    plt.scatter(y_true, y_pred, alpha= 0.3, s= 12)

    min_value = min(y_true.min(),y_pred.min())
    max_value = max(y_true.max(),y_pred.max())

    plt.plot([min_value, max_value],[min_value,max_value],linestyle= "--", linewidth= 2, color= "red", label= "Perfect Prediction")

    plt.title(f"Actual VS Predicted RUL\n{model_name}", fontsize= 16, fontweight= "bold")
    plt.xlabel("Actual RUL", fontsize= 13, fontweight= "bold")
    plt.ylabel("Predicted RUL", fontsize= 13, fontweight= "bold")

    plt.xticks(fontsize= 11)
    plt.yticks(fontsize= 11)

    plt.grid(alpha= 0.3)
    plt.legend()
    plt.tight_layout()

    save_figure(
    figure_name= model_name.lower().replace(" ","_"),
    subfolder="actual_vs_predicted",
    )

    plt.show()
    plt.close()


    logger.info(f"Actual VS Predicted RUL plot generated for {model_name}")
