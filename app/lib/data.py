"""Cached data access for the Streamlit dashboard."""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path("data/01_raw/dataset.db")
TABLE = "airline_passenger_satisfaction"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the full airline dataset from SQLite (cached, runs once)."""
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql(f"SELECT * FROM {TABLE}", conn)
