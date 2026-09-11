import matplotlib

matplotlib.use("Agg")

from src.data.ingestion import load_training_data
from src.features.feature_engineering import prepare_training_df
from src.features.preprocessing import prepare_training_data
from src.inference.error_analysis import run_error_analysis
from src.inference.evaluate_test_predictions import run_test_evaluation
from src.models.ablation import (
    create_ablation_report,
    plot_ablation_comparison,
    save_ablation_report,
)
from src.models.evaluate import compare_models, print_model_comparison, save_model_comparison
from src.models.workflow import run_all_models
from src.utils.logger import logger
from src.visualization.eda import run_eda


BASELINE_MODELS = ["linear_regression", "random_forest", "xgboost"]


def run_experiment(df, include_cycle: bool, models_to_run: list[str]) -> dict:
    """Prepare one feature set and run the requested models."""
    feature_set = "with cycle" if include_cycle else "without cycle"
    logger.info(f"Running experiment: {feature_set}")

    x_train, x_test, y_train, y_test, train_groups = prepare_training_data(
        df=df,
        include_cycle=include_cycle,
    )

    return run_all_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_groups=train_groups,
        models_to_run=models_to_run,
    )


def main() -> None:
    logger.info("Starting project pipeline")

    raw_df = load_training_data()
    feature_df = prepare_training_df(raw_df)

    run_eda(feature_df)

    results_with_cycle = run_experiment(feature_df, True, BASELINE_MODELS)
    results_without_cycle = run_experiment(feature_df, False, BASELINE_MODELS)

    ablation_df = create_ablation_report(results_with_cycle, results_without_cycle)
    save_ablation_report(ablation_df)
    plot_ablation_comparison(ablation_df)

    optimized_results = run_experiment(
        feature_df,
        True,
        ["optimized_xgboost"],
    )

    final_results = {**results_with_cycle, **optimized_results}
    comparison_df = compare_models(final_results)
    print_model_comparison(comparison_df)
    save_model_comparison(comparison_df)

    test_metrics = run_test_evaluation()
    print("\nNASA CMAPSS unseen-test results")
    print(test_metrics.to_string(index=False))

    error_summary = run_error_analysis()
    print("\nNASA test error summary")
    print(error_summary.to_string(index=False))

    logger.info("Project pipeline completed")


if __name__ == "__main__":
    main()
