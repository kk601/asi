"""Nodes for the data_processing pipeline."""

import logging
import sqlite3
from typing import Any, Dict, Tuple

import pandas as pd
import wandb
import os
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

load_dotenv()

logger = logging.getLogger(__name__)


def preprocess(db_path: str, parameters: Dict[str, Any]) -> pd.DataFrame:
    """Load data from SQLite, clean it and encode categorical columns."""
    db_path = db_path.strip()
    query = "SELECT * FROM airline_passenger_satisfaction"

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)

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

    target = parameters["target_column"].strip().lower().replace(" ", "_")

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        if col != target:
            df[col] = df[col].astype("category").cat.codes

    if df[target].dtype == "object":
        df[target] = df[target].astype("category").cat.codes

    logger.info("Preprocessing finished. Final shape: %s", df.shape)
    return df


def split_data(
    data: pd.DataFrame, parameters: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into train, validation and test subsets."""
    target = parameters["target_column"].strip().lower().replace(" ", "_")
    split_params = parameters["split"]

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
    X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]
) -> RandomForestClassifier:
    """Train a RandomForestClassifier."""
    model_params = parameters["model"]

    model = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        random_state=model_params["random_state"],
    )
    model.fit(X_train, y_train)

    logger.info("Model training finished")
    return model


def evaluate_and_log(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate the model on validation data and log results to Weights & Biases."""

    # Initialize Weights & Biases
    run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "asi-airline"),
            entity=os.getenv("WANDB_ENTITY"),
            name=f"rf-n{parameters['model']['n_estimators']}-default",
            config={
                "model_type": "RandomForest",
                "n_estimators": parameters["model"]["n_estimators"],
                "random_state": parameters["model"]["random_state"],
                "test_size":    parameters["split"]["test_size"],
            },
            tags=["baseline", "sklearn"],
        )

    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
    }

    # Log metrics to Weights & Biases
    wandb.log(metrics)

    # Log feature importances if available
    if hasattr(model, "feature_importances_"):
        wandb.sklearn.plot_feature_importances(
            model, feature_names=list(X_val.columns)
        )

    # Log the model as an artifact
    artifact = wandb.Artifact(
        name="baseline-model",
        type="model",
        description=f"RandomForest n={parameters['model']['n_estimators']}",
    )
    artifact.add_file("data/06_models/baseline_model.pkl")
    wandb.log_artifact(artifact)

    # Finish the run
    wandb.finish()

    logger.info(
        "Run finished and logged to Weights & Biases. Validation metrics -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    return metrics
