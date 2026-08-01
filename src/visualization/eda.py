from __future__ import annotations

import pandas as pd

from src.visualization.overview import (
    dataset_overview,
    analyze_missing_values,
)

from src.visualization.distributions import (
    engine_lifetime_analysis,
)

from src.visualization.sensor_analysis import (
    plot_sensor_trend,
)

from src.visualization.correlations import (
    sensor_correlation_with_rul,
    plot_top_feature_correlations,
    compare_dataset_correlations,
    plot_feature_consistency_heatmap,
    summarize_feature_consistency,
    plot_feature_correlation_heatmap

)

from src.utils.logger import logger


def run_eda(df: pd.DataFrame) -> None:
    logger.info("Starting EDA")

    dataset_overview(df)

    analyze_missing_values(df)

    engine_lifetime_analysis(df)

    plot_sensor_trend(
        df,
        dataset_id="FD001",
        unit_id=1,
        sensor="sensor_11",
    )

    correlation_df = sensor_correlation_with_rul(df)

    plot_top_feature_correlations(correlation_df)

    comparison_df = compare_dataset_correlations(df)

    plot_feature_consistency_heatmap(comparison_df)

    summarize_feature_consistency(comparison_df)

    for dataset in sorted(df["dataset_id"].unique()):
        plot_feature_correlation_heatmap(df,dataset)


    logger.info("EDA completed.")


    

