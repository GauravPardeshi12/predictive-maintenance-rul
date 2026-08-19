from src.data.ingestion import load_training_data
from src.features.feature_engineering import prepare_training_df
from src.visualization.eda import run_eda
from src.models.workflow import run_all_models
from src.features.preprocessing import prepare_training_data
from src.models.evaluate import compare_models, print_model_comparison, save_model_comparison
from src.models.tuning.xgboost_optuna import optimize_xgboost
def main():

    df = load_training_data()

    df = prepare_training_df(df)

    # run_eda(df)

    x_train, x_test, y_train, y_test, train_groups = prepare_training_data(df)

    # results = run_all_models(
    #     x_train,
    #     x_test,
    #     y_train,
    #     y_test,
    # )

    tuning_results = optimize_xgboost(
    x_train=x_train,
    y_train=y_train,
    groups=train_groups,
    n_trials=30,
)

    print("\n" + "=" * 60)
    print("XGBOOST OPTUNA RESULTS")
    print("=" * 60)

    print(f"Best Validation RMSE: {tuning_results['best_rmse']:.4f}")

    print("\nBest Parameters:")

    for parameter, value in tuning_results["best_params"].items():
        print(f"{parameter}: {value}")

    # comparison_df = compare_models(results= results)

    # print_model_comparison(comparison_df= comparison_df)

    # save_model_comparison(comparison_df)
    

if __name__ == "__main__":
    main()
