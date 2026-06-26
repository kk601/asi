"""Strona Predykcja — formularz 22 cech -> POST /predict."""

import requests
import streamlit as st

from lib import api_client

st.title("Predykcja satysfakcji")

GENDER = ["Female", "Male"]
CUSTOMER_TYPE = ["Returning", "First-time"]
TRAVEL = ["Business", "Personal"]
TRAVEL_CLASS = ["Business", "Economy", "Economy Plus"]
SURVEY = [
    "Departure and Arrival Time Convenience", "Ease of Online Booking",
    "Check-in Service", "Online Boarding", "Gate Location", "On-board Service",
    "Seat Comfort", "Leg Room Service", "Cleanliness", "Food and Drink",
    "In-flight Service", "In-flight Wifi Service", "In-flight Entertainment",
    "Baggage Handling",
]
D = api_client.DEFAULT_PASSENGER

st.subheader("Pasażer")
c1, c2, c3 = st.columns(3)
gender = c1.selectbox("Gender", GENDER, index=GENDER.index(D["Gender"]))
customer = c2.selectbox(
    "Customer Type", CUSTOMER_TYPE, index=CUSTOMER_TYPE.index(D["Customer Type"])
)
age = c3.number_input("Age", 1, 120, D["Age"])
travel = c1.selectbox(
    "Type of Travel", TRAVEL, index=TRAVEL.index(D["Type of Travel"])
)
travel_class = c2.selectbox(
    "Class", TRAVEL_CLASS, index=TRAVEL_CLASS.index(D["Class"])
)

st.subheader("Lot")
l1, l2, l3 = st.columns(3)
distance = l1.number_input("Flight Distance", 0, 10000, D["Flight Distance"])
dep_delay = l2.number_input("Departure Delay", 0, 2000, D["Departure Delay"])
arr_delay = l3.number_input("Arrival Delay", 0, 2000, D["Arrival Delay"])

st.subheader("Oceny ankietowe (0–5)")
survey_values = {}
cols = st.columns(2)
for i, field in enumerate(SURVEY):
    survey_values[field] = cols[i % 2].slider(field, 0, 5, int(D[field]))

if st.button("Przewiduj", type="primary"):
    payload = {
        "Gender": gender,
        "Customer Type": customer,
        "Age": age,
        "Type of Travel": travel,
        "Class": travel_class,
        "Flight Distance": distance,
        "Departure Delay": dep_delay,
        "Arrival Delay": arr_delay,
        **survey_values,
    }
    try:
        response = api_client.predict(payload)
        if response.status_code == 200:
            data = response.json()
            st.success(
                f"Predykcja: **{data['prediction_label']}** "
                f"(kod {data['prediction']})"
            )
            st.caption(f"Model: {data['model']}")
        elif response.status_code == 422:
            st.error("Błędne dane wejściowe (walidacja Pydantic).")
            st.json(response.json())
        elif response.status_code == 503:
            st.error(
                "API działa, ale model nie jest załadowany. "
                "Uruchom kedro run i zrestartuj API."
            )
        else:
            st.error(f"Błąd API ({response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Nie można połączyć się z API pod {api_client.API_URL}. "
            "Czy uvicorn api.main:app jest uruchomione?"
        )
