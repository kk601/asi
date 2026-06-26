"""Strona startowa — status API."""

import streamlit as st

from lib import api_client

st.title("Aplikacja satysfakcji pasażerów")
st.markdown(
    "Dashboard satysfakcji pasażerów linii lotniczych. "
    "Użyj menu po lewej: **Predykcja**, **Dane**, **Dane syntetyczne**."
)

st.subheader("Status API")
status = api_client.health()
if status.get("model_loaded"):
    st.success(f"API działa — model: {status.get('model_name', '?')}")
elif status.get("status") == "unreachable":
    st.error(
        f"API niedostępne pod {api_client.API_URL}. "
        "Uruchom: uvicorn api.main:app"
    )
else:
    st.warning(
        "API działa, ale model nie jest załadowany. "
        "Uruchom kedro run i zrestartuj API."
    )
