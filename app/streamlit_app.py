"""Aplikacja satysfakcji pasażerów — Streamlit dashboard (nawigacja)."""

import streamlit as st

st.set_page_config(page_title="Aplikacja satysfakcji pasażerów", layout="wide")

navigation = st.navigation(
    [
        st.Page("views/home.py", title="Aplikacja satysfakcji pasażerów", default=True),
        st.Page("views/prediction.py", title="Predykcja"),
        st.Page("views/data.py", title="Dane"),
        st.Page("views/synthetic_data.py", title="Dane syntetyczne"),
    ]
)
navigation.run()
