"""Nodes for the AutoML pipeline using AutoGluon."""

import logging
import os
import shutil
from typing import Any, Dict

import pandas as pd
import wandb
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

load_dotenv()

logger = logging.getLogger(__name__)

AUTOML_MODEL_PATH = "data/06_models/autogluon"


def _normalize_target_name(target_column: str) -> str:
    """Return target column name matching the preprocessing convention."""
    return target_column.strip().lower().replace(" ", "_")


def train_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_column: str,
    automl_params: Dict[str, Any],
) -> TabularPredictor:
    """Train AutoGluon TabularPredictor on the training dataset.

    Args:
        X_train: Training features.
        y_train: Training labels.
        target_column: Name of the target column from parameters.yml.
        automl_params: AutoGluon configuration with presets, time_limit and eval_metric.

    Returns:
        Trained AutoGluon TabularPredictor.
    """
    target = _normalize_target_name(target_column)

    X_train = X_train.copy()
    X_train.columns = [f"{column}" for column in X_train.columns]

    y_train = y_train.copy()
    y_train.name = target

    train_data = pd.concat([X_train, y_train], axis=1)

    if os.path.exists(AUTOML_MODEL_PATH):
        shutil.rmtree(AUTOML_MODEL_PATH)

    logger.info(
        "AutoGluon training started: presets=%s, time_limit=%s, eval_metric=%s",
        automl_params["presets"],
        automl_params["time_limit"],
        automl_params["eval_metric"],
    )

    predictor = TabularPredictor(
        label=target,
        eval_metric=automl_params["eval_metric"],
        path=AUTOML_MODEL_PATH,
    ).fit(
        train_data,
        presets=automl_params["presets"],
        time_limit=automl_params["time_limit"],
        verbosity=1,
    )

    logger.info("AutoGluon training finished")
    return predictor


def evaluate_automl(
    predictor: TabularPredictor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    automl_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate AutoGluon on validation data and log metrics to Weights & Biases.

    Logs the best model metrics and a full AutoGluon leaderboard as a W&B Table.

    Args:
        predictor: Trained AutoGluon TabularPredictor.
        X_val: Validation features.
        y_val: Validation labels.
        automl_params: AutoGluon configuration with presets, time_limit and eval_metric.

    Returns:
        Dictionary with validation metrics and AutoGluon run metadata.
    """
    X_val = X_val.copy()
    X_val.columns = [f"{column}" for column in X_val.columns]

    y_pred = predictor.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred, zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, zero_division=0)),
    }

    val_data = X_val.copy()
    val_data[predictor.label] = y_val

    leaderboard = predictor.leaderboard(data=val_data, silent=True)

    best_model_name = str(leaderboard.iloc[0]["model"])
    best_score_raw = float(leaderboard.iloc[0]["score_val"])

    eval_metric = automl_params["eval_metric"]
    best_metric_value = metrics.get(eval_metric, best_score_raw)

    metrics.update(
        {
            "best_model": best_model_name,
            eval_metric: float(best_metric_value),
            "best_score_val": best_score_raw,
            "n_models_trained": int(len(leaderboard)),
            "presets": automl_params["presets"],
            "time_limit": int(automl_params["time_limit"]),
            "eval_metric": eval_metric,
        }
    )
    # --- W&B: inicjalizacja ---
    with wandb.init(
            project=os.getenv("WANDB_PROJECT", "asi-airline"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"automl-{automl_params['presets']}-{automl_params['time_limit']}s",
            config={
                "model_type": "AutoGluon",
                "presets": automl_params["presets"],
                "time_limit": automl_params["time_limit"],
                "eval_metric": eval_metric,
                "best_model": best_model_name,
            },
            tags=["automl", "autogluon"],
    ):
        # --- W&B: metryka najlepszego modelu ---
        wandb.log({eval_metric: float(best_metric_value)})

        # Dodatkowo logujemy pozostałe metryki klasyfikacyjne,
        # żeby łatwo porównać AutoGluon z baseline.
        wandb.log(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "best_score_val": best_score_raw,
            }
        )

        # --- W&B: pełny leaderboard jako tabela ---
        # Widoczny w dashboardzie W&B w zakładce Tables, nie Charts
        leaderboard_columns = [
            column
            for column in ["model", "score_val", "pred_time_val", "fit_time"]
            if column in leaderboard.columns
        ]

        leaderboard_table = wandb.Table(
            dataframe=leaderboard[leaderboard_columns].reset_index(drop=True)
        )

        wandb.log({"leaderboard": leaderboard_table})

    logger.info(
        "AutoGluon validation metrics -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f; best_model=%s",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        best_model_name,
    )

    return metrics