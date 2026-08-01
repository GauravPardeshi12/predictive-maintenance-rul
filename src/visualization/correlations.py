from __future__ import annotations
import pandas as pd
from src.utils.logger import logger
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.helper import save_figure


def sensor_correlation_with_rul(
    df: pd.DataFrame, dataset_id: str | None = None
) -> pd.DataFrame:
    """
    Calculate the correlation between sensor-related features and RUL

    Parameters
    -----------
    df: pd.DataFrame
        Input Dataset

    Returns
    --------
    pd.DataFrame
        Sensor features ranked by correlation with RUL
    """
    logger.info("Calculating feature correlations with RUL.")

    if dataset_id is not None:
        logger.info(f"Analyzing {dataset_id}.")
        df = df[df["dataset_id"] == dataset_id]

    sensor_features = [column for column in df.columns if column.startswith("sensor_")]

    constant_features = [column for column in sensor_features if df[column].std() == 0]

    if constant_features:
        logger.warning(
            f"Detected {len(constant_features)} constant feature(s). "
            "These features were excluded from correlation analysis."
        )

        for feature in constant_features:
            logger.warning(f"   • {feature}")

    logger.info(
        f"Using {len(sensor_features)} sensor features for correlation analysis."
    )
    sensor_features = [
        column for column in sensor_features if column not in constant_features
    ]

    correlations = df[sensor_features].corrwith(df["RUL"]).dropna()

    correlation_df = correlations.to_frame("Correlation")

    correlation_df["abs_corr"] = correlation_df["Correlation"].abs()

    correlation_df = correlation_df.sort_values(by="abs_corr", ascending=False)

    logger.info("Sensor correlation analysis completed.")

    return correlation_df


def compare_dataset_correlations(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Compare the strongest correlated sensor feature across all NASA CMAPSS datasets.

    Parameters
    -----------
    df: pd.DataFrame
        Input Dataset

    Return
    ------------
        Summary of top correlated feature for each dataset.
    """
    logger.info("Comparing features correlation across dataset.")
    result = []

    for dataset in sorted(df["dataset_id"].unique()):
        correlation_df = sensor_correlation_with_rul(df, dataset)
        top_features = correlation_df.head(top_n)
        for feature, row in top_features.iterrows():
            result.append(
                {
                    "Dataset": dataset,
                    "Feature": feature,
                    "Correlation": row["Correlation"],
                }
            )
    comparison_df = pd.DataFrame(result)

    logger.info(
        f"Compared {comparison_df['Feature'].nunique()} unique features across "
        f"{comparison_df['Dataset'].nunique()} datasets."
    )
    return comparison_df


def plot_top_feature_correlations(
    correlation_df: pd.DataFrame, top_n: int = 10
) -> None:
    """
    Plot the top correlated features with RUL

    Parameters
    -----------
    correlation_df: pd.DataFrame
        Correlation DataFrame

    top_n: int
        Number of top features to display
    """
    logger.info("Plotting top correlated features.")

    top_features = correlation_df.head(top_n)
    top_features = top_features.iloc[::-1]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features.index, top_features["Correlation"])
    plt.axvline(x=0, color="black", linewidth=1)
    plt.xlabel("Correlation with RUL")
    plt.ylabel("Sensor features")
    plt.title("Top Correlated Featues")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    save_figure(figure_name="top_correlated_features", subfolder= "eda")
    plt.show()
    plt.close()

    logger.info("Top correlated feature plot generated.")


def plot_feature_consistency_heatmap(comparison_df: pd.DataFrame) -> None:
    """
    Plot a heatmap showing how the top correlated features vary across NASA CMAPSS datasets.

    Parameters
    ------------
    comparison_df: pd.DataFrame
        Output from compare_dataset_correlations().
    """

    logger.info("Generating feature consistency heatmap.")

    heatmap_df = comparison_df.pivot_table(
        index="Feature", columns="Dataset", values="Correlation"
    )

    feature_order = (
        comparison_df.groupby("Feature")["Dataset"]
        .nunique()
        .sort_values(ascending=False)
        .index
    )
    heatmap_df = heatmap_df.loc[feature_order]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="coolwarm",
        linewidth=0.5,
        fmt=".2f",
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.9},
    )
    plt.title(
        "Feature correlations across NASA CMAPSS Datasets",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel("Dataset", fontsize=14, fontweight="bold")
    plt.ylabel("Feature", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12, fontweight="bold")
    plt.yticks(fontsize=11)
    plt.tight_layout()

    save_figure(
    figure_name="feature_consistency_heatmap",
    subfolder="eda",
    )

    plt.show()
    plt.close()

    logger.info("Feature consistency heatmap generated.")


def summarize_feature_consistency(comparison_df: pd.DataFrame) -> None:
    """
    Summarizing Feature consistency

    Parameters
    ------------
    comparison_df: pd.DataFrame
        Output from compare_dataset_correlations().
    """
    logger.info("Generating feature consistency summary.")
    print("\n")
    print("=" * 70)
    print("FEATURE CONSISTENCY SUMMARY")
    print("=" * 70)

    feature_counts = (
        comparison_df.groupby("Feature")["Dataset"]
        .nunique()
        .sort_values(ascending=False)
    )
    print("\nConsistent Feature (Present in all datasets)\n")
    for feature, count in feature_counts.items():
        if count == 4:
            print(f"✓ {feature}")

    print("\nModerately Consistent (Present in 2-3 datasets)\n")
    for feature, count in feature_counts.items():
        if 2 <= count < 4:
            print(f"• {feature} ({count} datasets)")

    print("\nDataset Specific Features\n")
    for feature, count in feature_counts.items():
        if count == 1:
            print(f"- {feature}")

    logger.info("Feature consistency summary generated.")


def plot_feature_correlation_heatmap(
    df: pd.DataFrame, dataset_id: str, top_n: int = 10
) -> None:
    """
    Plot the correlation heatmap among the top correlated
    features for a single NASA CMAPSS dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.

    dataset_id : str
        Dataset to analyze (FD001, FD002, FD003, FD004)

    top_n : int
        Number of top correlated features to include.
    """

    logger.info(f"Generating feature-feature correlation heatmap for {dataset_id}.")

    correlation_df = sensor_correlation_with_rul(df=df, dataset_id=dataset_id)

    if correlation_df.empty:
        logger.warning(f"No correlated features found for {dataset_id}")
        return

    top_features = correlation_df.head(top_n).index.to_list()

    logger.info(f"Using Top {len(top_features)} correlated feature")

    dataset_df = df.query("dataset_id == @dataset_id")

    feature_corr = dataset_df[top_features].corr()

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        feature_corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.8,
        linecolor="white",
        annot_kws={
            "fontsize": 9,
            "fontweight": "bold",
        },
        cbar_kws={
            "label": "Pearson Correlation",
            "shrink": 0.9,
        },
    )

    plt.title(
        f"Top Feature Correlation Heatmap ({dataset_id})",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    plt.xlabel(
        "Feature",
        fontsize=13,
        fontweight="bold",
    )

    plt.ylabel(
        "Feature",
        fontsize=13,
        fontweight="bold",
    )

    plt.xticks(
        rotation=45,
        ha="right",
        fontsize=10,
    )

    plt.yticks(
        rotation=0,
        fontsize=10,
    )

    plt.tight_layout()

    plt.show()

    logger.info(f"Feature correlation heatmap generated for {dataset_id}.")
