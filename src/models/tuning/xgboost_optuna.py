from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

from src.utils.config import config
from src.utils.logger import logger


RANDOM_SEED = config["project"]["random_seed"]
VALIDATION_SIZE = config["training"]["test_size"]
RUL_CAP = config["training"]["rul_cap"]
N_TRIALS = config["training"]["optuna_trials"]


def create_validation_split(x_train, y_train, groups):
    """Create an engine-wise validation split and keep final-cycle rows for scoring."""
    groups = groups.astype(str)
    train_indices = []
    validation_indices = []
    rng = np.random.default_rng(RANDOM_SEED)

    for dataset_id in sorted(groups.str.split("_").str[0].unique()):
        mask = groups.str.startswith(f"{dataset_id}_")
        indices = np.flatnonzero(mask.to_numpy())
        dataset_groups = groups.iloc[indices]
        engines = dataset_groups.drop_duplicates().to_numpy()

        validation_count = max(1, int(len(engines) * VALIDATION_SIZE))
        validation_engines = set(rng.permutation(engines)[:validation_count])
        validation_mask = dataset_groups.isin(validation_engines).to_numpy()

        validation_indices.extend(indices[validation_mask].tolist())
        train_indices.extend(indices[~validation_mask].tolist())

    x_tune_train = x_train.iloc[train_indices]
    x_validation = x_train.iloc[validation_indices]
    y_tune_train = y_train.iloc[train_indices]
    y_validation = y_train.iloc[validation_indices]

    validation_groups = groups.iloc[validation_indices].reset_index(drop=True)
    final_positions = (
        x_validation.reset_index(drop=True)
        .assign(_engine=validation_groups)
        .groupby("_engine", sort=False)["cycle"]
        .idxmax()
        .to_numpy()
    )

    return (
        x_tune_train,
        x_validation.iloc[final_positions],
        y_tune_train,
        y_validation.iloc[final_positions],
    )


def optimize_xgboost(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    n_trials: int | None = None,
) -> dict:
    """Tune XGBoost on engine-final-cycle validation rows."""
    (
        x_tune_train,
        x_validation,
        y_tune_train,
        y_validation,
    ) = create_validation_split(x_train, y_train, groups)

    capped_y_train = y_tune_train.clip(upper=RUL_CAP)
    n_trials = n_trials or N_TRIALS

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 250, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.06, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 5.0, log=True),
            "objective": "reg:squarederror",
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
            "verbosity": 0,
        }

        model = XGBRegressor(**params)
        model.fit(x_tune_train, capped_y_train)
        predictions = model.predict(x_validation)
        return root_mean_squared_error(y_validation, predictions)

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    logger.info(f"Running {n_trials} Optuna trials with RUL cap={RUL_CAP}")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best validation RMSE: {study.best_value:.4f}")
    return {
        "best_params": study.best_params,
        "best_rmse": study.best_value,
        "study": study,
    }
