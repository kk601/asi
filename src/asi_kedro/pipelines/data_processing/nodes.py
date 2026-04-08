"""Nodes for the data_processing pipeline."""

import logging
import sqlite3
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

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


def evaluate_model(
    model: RandomForestClassifier, X_val: pd.DataFrame, y_val: pd.Series
) -> Dict[str, float]:
    """Evaluate the model on validation data."""
    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
    }

    logger.info(
        "Validation metrics -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )
    return metrics
