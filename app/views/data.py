"""Strona Dane — podgląd z SQLite + statystyki + wykresy."""

import streamlit as st

from lib.data import load_data

st.title("Dane")

df = load_data()
st.write(f"Liczba rekordów: {len(df)}")
st.dataframe(df.head(100), use_container_width=True)

st.subheader("Statystyki opisowe (kolumny numeryczne)")
st.dataframe(df.describe(), use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Rozkład satysfakcji")
    st.bar_chart(df["Satisfaction"].value_counts())
with col_b:
    st.subheader("Rozkład klas")
    st.bar_chart(df["Class"].value_counts())

st.subheader("Histogram wybranej kolumny numerycznej")
num_cols = df.select_dtypes("number").columns.tolist()
default_idx = num_cols.index("Age") if "Age" in num_cols else 0
column = st.selectbox("Kolumna", num_cols, index=default_idx)
st.bar_chart(df[column].value_counts().sort_index())
