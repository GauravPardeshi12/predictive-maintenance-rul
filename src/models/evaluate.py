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

def plot_feature_importance(model, feature_names, model_name: str, top_n: int = 20) -> pd.DataFrame:
    """
    Plot and return the most important features for a tree-based model.

    Parameters
    ----------
    model : tree-based regressor
        Trained model containing feature_importances_.

    feature_names : iterable
        Names of the model input features.

    model_name : str
        Name of the trained model.

    top_n : int, default=20
        Number of top features to display.

    Returns
    -------
    pd.DataFrame
        Feature importance ranked from highest to lowest.
    """

    logger.info(f"Generating feature importance for {model_name}")

    if not hasattr(model, "feature_importances_"):
        raise AttributeError(f"{model_name} does not provide feature_importances_.")

    importance_df = pd.DataFrame({"Feature": feature_names,
                                  "Importance": model.feature_importances_})

    importance_df = importance_df.sort_values(by= "Importance", ascending= False).reset_index(drop= True)
    top_features = importance_df.head(top_n).sort_values(by= "Importance", ascending= True)

    plt.figure(figsize= (10,8))

    plt.barh(top_features["Feature"], top_features["Importance"])

    plt.title(
        f"Top {top_n} Feature Importance\n({model_name})",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel(
        "Feature Importance",
        fontsize=13,
        fontweight="bold",
    )

    plt.ylabel(
        "Feature",
        fontsize=13,
        fontweight="bold",
    )

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=10)

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    save_figure(
        figure_name=f"{model_name}_feature_importance",
        subfolder="feature_importance",
    )

    plt.show()
    plt.close()

    logger.info(
        f"Feature importance plot generated for {model_name}."
    )

    return importance_df


def save_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str,
) -> Path:
    """
    Save feature importance results as a CSV report.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance DataFrame.

    model_name : str
        Name of the model.

    Returns
    -------
    Path
        Path to the saved feature importance report.
    """

    logger.info(
        f"Saving feature importance report for {model_name}."
    )

    metrics_dir = Path(
        config["paths"]["metrics"]
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_name = (
        f"{model_name.lower().replace(' ', '_')}"
        "_feature_importance.csv"
    )

    report_path = metrics_dir / file_name

    importance_df.to_csv(
        report_path,
        index=False,
    )

    logger.info(
        f"Feature importance report saved to {report_path}."
    )

    return report_path


def calculate_residuals(y_true: pd.Series, y_pred: np.ndarray) ->np.ndarray:
    """
    Calculate prediction residuals.

    Residual = Actual RUL - Predicted RUL.
    """

    logger.info("Calculating prediction residuals.")

    residuals = np.asarray(y_true) - np.asarray(y_pred)

    logger.info("Prediction residuals calculated.")
    return residuals


def plot_residual_distribution(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """
    Plot the distribution of model residuals.

    Residual = Actual RUL - Predicted RUL.
    """

    logger.info(
        f"Generating residual distribution for {model_name}."
    )

    residuals = calculate_residuals(
        y_true=y_true,
        y_pred=y_pred,
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        residuals,
        bins=40,
        alpha=0.7,
    )

    plt.axvline(
        x=0,
        linestyle="--",
        linewidth=2,
        color="red",
        label="Zero Error",
    )

    plt.title(
        f"Residual Distribution\n({model_name})",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel(
        "Residual (Actual - Predicted)",
        fontsize=13,
        fontweight="bold",
    )

    plt.ylabel(
        "Frequency",
        fontsize=13,
        fontweight="bold",
    )

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    save_figure(
        figure_name=f"{model_name}_residual_distribution",
        subfolder="residuals",
    )

    plt.show()
    plt.close()

    logger.info(
        f"Residual distribution generated for {model_name}."
    )


def plot_residuals_vs_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """
    Plot residuals against predicted RUL values.
    """

    logger.info(
        f"Generating residual vs prediction plot for {model_name}."
    )

    residuals = calculate_residuals(
        y_true=y_true,
        y_pred=y_pred,
    )

    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.4,
        s=12,
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
        color="red",
        label="Zero Error",
    )

    plt.title(
        f"Residuals vs Predicted RUL\n({model_name})",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel(
        "Predicted RUL",
        fontsize=13,
        fontweight="bold",
    )

    plt.ylabel(
        "Residual (Actual - Predicted)",
        fontsize=13,
        fontweight="bold",
    )

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    save_figure(
        figure_name=f"{model_name}_residuals_vs_predictions",
        subfolder="residuals",
    )

    plt.show()
    plt.close()

    logger.info(
        f"Residual vs prediction plot generated for {model_name}."
    )



