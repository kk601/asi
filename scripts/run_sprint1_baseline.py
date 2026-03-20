#!/usr/bin/env python
"""Train and evaluate the sprint 1 baseline model from SQLite."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TABLE_NAME = "airline_passenger_satisfaction"
NUMERIC_COLUMNS = [
    "ID",
    "Age",
    "Flight Distance",
    "Departure Delay",
    "Arrival Delay",
    "Departure and Arrival Time Convenience",
    "Ease of Online Booking",
    "Check-in Service",
    "Online Boarding",
    "Gate Location",
    "On-board Service",
    "Seat Comfort",
    "Leg Room Service",
    "Cleanliness",
    "Food and Drink",
    "In-flight Service",
    "In-flight Wifi Service",
    "In-flight Entertainment",
    "Baggage Handling",
]


def resolve_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def load_dataset(database_path: Path) -> pd.DataFrame:
    """Load the SQLite table and normalize column types."""
    query = f"SELECT * FROM {TABLE_NAME}"
    with sqlite3.connect(database_path) as connection:
        dataset = pd.read_sql_query(query, connection)

    dataset = dataset.rename(columns=lambda column: column.strip())
    dataset = dataset.replace({"": pd.NA, "NA": pd.NA, "NaN": pd.NA})

    for column in NUMERIC_COLUMNS:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")

    for column in dataset.columns:
        if column not in NUMERIC_COLUMNS:
            dataset[column] = dataset[column].astype("string")

    return dataset


def split_features_target(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for model training."""
    deduplicated = dataset.drop_duplicates().copy()
    deduplicated["target"] = (deduplicated["Satisfaction"].str.lower() == "satisfied").astype(int)

    features = deduplicated.drop(columns=["Satisfaction", "target", "ID"])
    target = deduplicated["target"]
    return features, target


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    """Create the preprocessing and baseline model pipeline."""
    categorical_features = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_features = [column for column in features.columns if column not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float | int]:
    """Compute evaluation metrics for one dataset split."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(target, predictions)), 4),
        "precision": round(float(precision_score(target, predictions)), 4),
        "recall": round(float(recall_score(target, predictions)), 4),
        "f1": round(float(f1_score(target, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
        "size": int(len(target)),
    }


def main() -> None:
    """Run the baseline training pipeline and save metrics."""
    project_root = resolve_project_root()
    database_path = project_root / os.getenv("DATABASE_PATH", "data/01_raw/dataset.db")
    metrics_path = project_root / os.getenv(
        "METRICS_PATH",
        "reports/metrics/baseline_metrics.json",
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if not database_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {database_path}")

    dataset = load_dataset(database_path)
    features, target = split_features_target(dataset)
    duplicate_rows = int(dataset.duplicated().sum())

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=42,
        stratify=target,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    baseline_model = build_pipeline(features)
    baseline_model.fit(x_train, y_train)

    metrics = {
        "model": "RandomForestClassifier",
        "validation": evaluate(baseline_model, x_val, y_val),
        "test": evaluate(baseline_model, x_test, y_test),
        "dataset": {
            "rows": int(len(dataset)),
            "rows_after_deduplication": int(len(features)),
            "features": int(features.shape[1]),
            "train_size": int(len(x_train)),
            "validation_size": int(len(x_val)),
            "test_size": int(len(x_test)),
            "duplicates_removed": duplicate_rows,
            "missing_values_total": int(dataset.isna().sum().sum()),
        },
    }

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
