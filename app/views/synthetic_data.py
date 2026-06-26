"""Strona Dane syntetyczne — interaktywne SDV + porównanie real vs syntetyczne."""

import pandas as pd
import streamlit as st

from lib.data import load_data
from lib.synth import fit_synthesizer

st.title("Dane syntetyczne (SDV)")

df = load_data()
n_samples = st.number_input(
    "Liczba rekordów do wygenerowania", 100, 10000, 1000, step=100
)

if st.button("Generuj dane syntetyczne"):
    with st.spinner("Trenowanie synthesizera i generowanie…"):
        synthesizer = fit_synthesizer(df)
        st.session_state["synthetic"] = synthesizer.sample(num_rows=int(n_samples))

if "synthetic" in st.session_state:
    synthetic = st.session_state["synthetic"]
    st.success(f"Wygenerowano {len(synthetic)} rekordów.")

    real_no_id = df.drop(columns=["ID", "id"], errors="ignore")
    left, right = st.columns(2)
    with left:
        st.subheader("Oryginał (statystyki)")
        st.dataframe(real_no_id.describe())
    with right:
        st.subheader("Syntetyczne (statystyki)")
        st.dataframe(synthetic.describe())

    st.subheader("Porównanie rozkładu kolumny")
    column = st.selectbox("Kolumna", synthetic.columns.tolist())
    real_counts = real_no_id[column].value_counts().sort_index().rename("real")
    synth_counts = synthetic[column].value_counts().sort_index().rename("syntetyczne")
    comparison = pd.concat([real_counts, synth_counts], axis=1).fillna(0)
    st.bar_chart(comparison)
