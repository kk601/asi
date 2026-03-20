#!/usr/bin/env python
"""Load the Airline Passenger Satisfaction CSV into SQLite."""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from pathlib import Path

TABLE_NAME = "airline_passenger_satisfaction"
COLUMNS = [
    "ID",
    "Gender",
    "Age",
    "Customer Type",
    "Type of Travel",
    "Class",
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
    "Satisfaction",
]

CREATE_TABLE_SQL = f"""
CREATE TABLE {TABLE_NAME} (
    ID INTEGER,
    Gender TEXT,
    Age INTEGER,
    [Customer Type] TEXT,
    [Type of Travel] TEXT,
    Class TEXT,
    [Flight Distance] INTEGER,
    [Departure Delay] INTEGER,
    [Arrival Delay] INTEGER,
    [Departure and Arrival Time Convenience] INTEGER,
    [Ease of Online Booking] INTEGER,
    [Check-in Service] INTEGER,
    [Online Boarding] INTEGER,
    [Gate Location] INTEGER,
    [On-board Service] INTEGER,
    [Seat Comfort] INTEGER,
    [Leg Room Service] INTEGER,
    Cleanliness INTEGER,
    [Food and Drink] INTEGER,
    [In-flight Service] INTEGER,
    [In-flight Wifi Service] INTEGER,
    [In-flight Entertainment] INTEGER,
    [Baggage Handling] INTEGER,
    Satisfaction TEXT
)
"""


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Load Airline Passenger Satisfaction data from CSV into SQLite."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to airline_passenger_satisfaction.csv",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(os.getenv("DATABASE_PATH", "data/01_raw/dataset.db")),
        help="Output SQLite database path",
    )
    return parser.parse_args()


def load_rows(source_path: Path) -> list[tuple[str, ...]]:
    """Read CSV rows in the expected column order."""
    with source_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = [column for column in COLUMNS if column not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return [tuple(row[column] for column in COLUMNS) for row in reader]


def write_sqlite(rows: list[tuple[str, ...]], database_path: Path) -> None:
    """Replace the SQLite table contents with CSV rows."""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        cursor.execute(CREATE_TABLE_SQL)
        placeholders = ",".join(["?"] * len(COLUMNS))
        cursor.executemany(
            f"INSERT INTO {TABLE_NAME} VALUES ({placeholders})",
            rows,
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_satisfaction ON {TABLE_NAME} (Satisfaction)"
        )
        connection.commit()


def main() -> None:
    """Run the import script."""
    args = parse_args()
    source_path = args.source.resolve()
    database_path = args.database.resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source CSV does not exist: {source_path}")

    rows = load_rows(source_path)
    write_sqlite(rows, database_path)
    print(f"Loaded {len(rows)} rows into {database_path}")


if __name__ == "__main__":
    main()
