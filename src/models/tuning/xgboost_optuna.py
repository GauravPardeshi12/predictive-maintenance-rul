from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.utils.logger import logger


def create_validation_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """
    Create a dataset-stratified engine-wise validation split.
    """

    logger.info(
        "Creating dataset-stratified engine-wise validation split."
    )

    groups = groups.astype(str)

    train_indices = []
    validation_indices = []

    dataset_ids = (
        groups.str.split("_")
        .str[0]
        .unique()
    )

    rng = np.random.default_rng(42)

    for dataset_id in sorted(dataset_ids):

        dataset_mask = groups.str.startswith(
            f"{dataset_id}_"
        )

        dataset_indices = np.flatnonzero(
            dataset_mask.to_numpy()
        )

        dataset_groups = groups.iloc[
            dataset_indices
        ]

        unique_engines = (
            dataset_groups
            .drop_duplicates()
            .to_numpy()
        )

        shuffled_engines = rng.permutation(
            unique_engines
        )

        validation_engine_count = max(
            1,
            int(len(unique_engines) * 0.20),
        )

        validation_engines = set(
            shuffled_engines[
                :validation_engine_count
            ]
        )

        validation_mask = dataset_groups.isin(
            validation_engines
        )

        current_validation_indices = (
            dataset_indices[
                validation_mask.to_numpy()
            ]
        )

        current_train_indices = (
            dataset_indices[
                ~validation_mask.to_numpy()
            ]
        )

        train_indices.extend(
            current_train_indices.tolist()
        )

        validation_indices.extend(
            current_validation_indices.tolist()
        )

        logger.info(
            f"{dataset_id}: "
            f"{len(unique_engines) - validation_engine_count} "
            f"training engines, "
            f"{validation_engine_count} validation engines."
        )

        logger.info(
            f"{dataset_id}: "
            f"{len(current_train_indices):,} training rows, "
            f"{len(current_validation_indices):,} validation rows."
        )

    train_indices = np.asarray(
        train_indices,
        dtype=int,
    )

    validation_indices = np.asarray(
        validation_indices,
        dtype=int,
    )

    logger.info(
        f"Total tuning rows: {len(train_indices):,}"
    )

    logger.info(
        f"Total validation rows: {len(validation_indices):,}"
    )

    return (
        x_train.iloc[train_indices],
        x_train.iloc[validation_indices],
        y_train.iloc[train_indices],
        y_train.iloc[validation_indices],
    )


def optimize_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    n_trials: int = 30,
) -> dict:
    """
    Optimize XGBoost hyperparameters using Optuna.
    """

    logger.info(
        "Starting XGBoost hyperparameter optimization."
    )

    (
        x_tune_train,
        x_validation,
        y_tune_train,
        y_validation,
    ) = create_validation_split(
        x_train=x_train,
        y_train=y_train,
        groups=groups,
    )

    logger.info(
        f"Tuning training shape: {x_tune_train.shape}"
    )

    logger.info(
        f"Validation shape: {x_validation.shape}"
    )

    def objective(trial: optuna.Trial) -> float:

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                200,
                800,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.15,
                log=True,
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                3,
                10,
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight",
                1,
                10,
            ),
            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
            "gamma": trial.suggest_float(
                "gamma",
                0.0,
                5.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha",
                1e-8,
                10.0,
                log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda",
                1e-3,
                10.0,
                log=True,
            ),
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }

        model = XGBRegressor(
            **params
        )

        model.fit(
            x_tune_train,
            y_tune_train,
        )

        predictions = model.predict(
            x_validation
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true=y_validation,
                y_pred=predictions,
            )
        )

        return rmse

    study = optuna.create_study(
        direction="minimize",
        study_name="xgboost_rul_optimization",
    )

    logger.info(
        f"Running {n_trials} Optuna trials."
    )

    study.optimize(
        objective,
        n_trials=n_trials,
    )

    logger.info(
        "XGBoost hyperparameter optimization completed."
    )

    logger.info(
        f"Best validation RMSE: "
        f"{study.best_value:.4f}"
    )

    logger.info(
        f"Best parameters: "
        f"{study.best_params}"
    )

    return {
        "best_params": study.best_params,
        "best_rmse": study.best_value,
        "study": study,
    }