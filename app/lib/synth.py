"""Cached SDV synthesizer for the Streamlit dashboard.

NOTE: the GaussianCopula fitting logic is intentionally duplicated here and in
src/asi_kedro/pipelines/synthetic/nodes.py (per the course instruction).
"""

import pandas as pd
import streamlit as st
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

ID_COLUMNS = ["ID", "id"]


@st.cache_resource
def fit_synthesizer(real_data: pd.DataFrame) -> GaussianCopulaSynthesizer:
    """Fit a GaussianCopula synthesizer once (cached) on data without ID."""
    data = real_data.drop(columns=ID_COLUMNS, errors="ignore")
    metadata = Metadata.detect_from_dataframe(data)
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(data)
    return synth
