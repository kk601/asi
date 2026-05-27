"""Nodes for the data_processing pipeline."""

import logging
import os
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

load_dotenv()

logger = logging.getLogger(__name__)


def preprocess(
    data: pd.DataFrame,
    target_column: str,
    split_params: Dict[str, Any],
) -> pd.DataFrame:
    """Clean dataset and encode categorical columns.

    Args:
        data: Raw dataframe loaded by the Kedro Data Catalog
            (``airline_raw`` → ``pandas.SQLTableDataset``).
        target_column: Name of the target column (e.g. ``"satisfaction"``).
        split_params: Sub-dict from parameters.yml; only ``random_state``
            is used here, as a seed for any non-deterministic step.

    Returns:
        Cleaned, fully numeric dataframe ready for ``split_data``.
    """
    np.random.seed(split_params.get("random_state", 42))

    df = data.copy()

    # Ujednolicenie nazw kolumn
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    logger.info("Columns after normalization: %s", list(df.columns))

    if "unnamed:_0" in df.columns:
        df = df.drop(columns=["unnamed:_0"])
    if "unnamed: 0" in df.columns:
        df = df.drop(columns=["unnamed: 0"])

    before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows", before - len(df))

    df = df.replace("", pd.NA)

    before = len(df)
    df = df.dropna()
    logger.info("Removed %d rows with missing values", before - len(df))

    target = target_column.strip().lower().replace(" ", "_")

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if col != target:
            df[col] = df[col].astype("category").cat.codes

    if df[target].dtype == "object":
        df[target] = df[target].astype("category").cat.codes

    logger.info("Preprocessing finished. Final shape: %s", df.shape)
    return df


def split_data(
    data: pd.DataFrame,
    target_column: str,
    split_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into train, validation and test subsets.

    Args:
        data: Output of ``preprocess`` (fully numeric dataframe).
        target_column: Name of the target column.
        split_params: ``{"test_size": float, "val_ratio": float, "random_state": int}``.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    data.columns = [f"{column}" for column in data.columns]

    target = target_column.strip().lower().replace(" ", "_")

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=split_params["val_ratio"],
        random_state=split_params["random_state"],
        stratify=y_temp,
    )

    logger.info(
        "Split sizes -> train: %d, val: %d, test: %d",
        len(X_train),
        len(X_val),
        len(X_test),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict[str, Any],
) -> RandomForestClassifier:
    """Train a RandomForestClassifier.

    Args:
        X_train: Training features.
        y_train: Training target.
        model_params: ``{"n_estimators": int, "max_depth": int|None, "random_state": int}``.

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        random_state=model_params["random_state"],
        max_depth=model_params["max_depth"],
    )
    model.fit(X_train, y_train)

    logger.info("Model training finished")
    return model


def evaluate_and_log(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_params: Dict[str, Any],
    split_params: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate the model on validation data and log results to Weights & Biases.

    Args:
        model: Trained scikit-learn classifier.
        X_val: Validation features.
        y_val: Validation target.
        model_params: Sub-dict ``{"n_estimators", "max_depth", "random_state"}``.
        split_params: Sub-dict ``{"test_size", "val_ratio", "random_state"}``.

    Returns:
        Dict with validation metrics (accuracy, precision, recall, f1).
    """
    # Inicjalizacja W&B przez context manager — gwarancja wandb.finish() przy wyjątku.
    # WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT odczytywane są z .env.
    with wandb.init(
        project=os.getenv("WANDB_PROJECT", "asi-airline"),
        entity=os.getenv("WANDB_ENTITY"),
        name=f"rf-n{model_params['n_estimators']}-d{model_params['max_depth']}",
        config={
            "model_type": "RandomForest",
            "n_estimators": model_params["n_estimators"],
            "max_depth": model_params["max_depth"],
            "random_state": model_params["random_state"],
            "test_size": split_params["test_size"],
        },
        tags=["baseline", "sklearn"],
    ) as run:
        y_pred = model.predict(X_val)

        # zero_division=0 chroni przed warningiem gdy klasa nie występuje w predykcji
        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
        }

        wandb.log(metrics)

        # Wandb wewnątrz plot_feature_importances konwertuje DataFrame do np.array,
        # co wywołuje sklearn warning "X does not have valid feature names".
        # Tłumimy tylko ten konkretny warning lokalnie, na czas wywołania.
        if hasattr(model, "feature_importances_"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names",
                    category=UserWarning,
                )
                wandb.sklearn.plot_feature_importances(
                    model, feature_names=list(X_val.columns)
                )

        artifact = wandb.Artifact(
            name="baseline-model",
            type="model",
            description=f"RandomForest n={model_params['n_estimators']}",
        )
        artifact.add_file("data/06_models/baseline_model.pkl")
        run.log_artifact(artifact)

    logger.info(
        "Run finished and logged to Weights & Biases. "
        "Validation metrics -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    return metrics
