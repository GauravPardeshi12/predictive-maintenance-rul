from src.data.ingestion import load_training_data
from src.features.feature_engineering import prepare_training_df
from src.visualization.eda import run_eda
from src.models.workflow import run_all_models
from src.features.preprocessing import prepare_training_data
from src.models.evaluate import compare_models, print_model_comparison, save_model_comparison

def main():

    df = load_training_data()

    df = prepare_training_df(df)

    run_eda(df)

    x_train, x_test, y_train, y_test, train_groups = prepare_training_data(df)

    results = run_all_models(
        x_train,
        x_test,
        y_train,
        y_test,
    )

    comparison_df = compare_models(results= results)

    print_model_comparison(comparison_df= comparison_df)

    save_model_comparison(comparison_df)
    

if __name__ == "__main__":
    main()
