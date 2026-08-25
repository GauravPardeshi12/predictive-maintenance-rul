import matplotlib

matplotlib.use("Agg")

from src.data.ingestion import load_training_data
from src.features.feature_engineering import prepare_training_df
from src.visualization.eda import run_eda
from src.features.preprocessing import prepare_training_data
from src.models.workflow import run_all_models
from src.models.evaluate import (
    compare_models,
    print_model_comparison,
    save_model_comparison,
)
from src.models.ablation import (
    create_ablation_report,
    save_ablation_report,
    plot_ablation_comparison,
)


def run_experiment(
    df,
    include_cycle: bool,
    models_to_run: list[str],
) -> dict:

    experiment_name = (
        "With Cycle"
        if include_cycle
        else "Without Cycle"
    )

    print("\n" + "=" * 70)
    print(f"RUNNING EXPERIMENT: {experiment_name}")
    print("=" * 70)

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_groups,
    ) = prepare_training_data(
        df=df,
        include_cycle=include_cycle,
    )

    results = run_all_models(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        train_groups=train_groups,
        models_to_run=models_to_run,
    )

    return results

def main():

    df = load_training_data()

    df = prepare_training_df(df)

    run_eda(df)

    baseline_models = [
        "linear_regression",
        "random_forest",
        "xgboost",
    ]

    results_with_cycle = run_experiment(
        df=df,
        include_cycle=True,
        models_to_run=baseline_models,
    )

    results_without_cycle = run_experiment(
        df=df,
        include_cycle=False,
        models_to_run=baseline_models,
    )

    ablation_df = create_ablation_report(
        results_with_cycle=results_with_cycle,
        results_without_cycle=results_without_cycle,
    )

    save_ablation_report(
        ablation_df=ablation_df
    )

    plot_ablation_comparison(
        ablation_df=ablation_df
    )

    optimized_results = run_experiment(
        df=df,
        include_cycle=True,
        models_to_run=[
            "optimized_xgboost",
        ],
    )

    final_results = {
    **results_with_cycle,
    **optimized_results,
    }

    final_comparison = compare_models(
        results=final_results
    )

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print_model_comparison(
        comparison_df=final_comparison
    )

    save_model_comparison(
        comparison_df=final_comparison
    )

if __name__ == "__main__":
    main()