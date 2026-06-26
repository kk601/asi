"""HTTP client for the Sprint 5 FastAPI prediction service."""

import os
from typing import Any, Dict

import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

# Default passenger so the prediction form works on first click (22 features).
DEFAULT_PASSENGER: Dict[str, Any] = {
    "Gender": "Female",
    "Customer Type": "Returning",
    "Age": 35,
    "Type of Travel": "Business",
    "Class": "Business",
    "Flight Distance": 1200,
    "Departure Delay": 10,
    "Arrival Delay": 5,
    "Departure and Arrival Time Convenience": 4,
    "Ease of Online Booking": 3,
    "Check-in Service": 4,
    "Online Boarding": 4,
    "Gate Location": 3,
    "On-board Service": 5,
    "Seat Comfort": 4,
    "Leg Room Service": 4,
    "Cleanliness": 5,
    "Food and Drink": 3,
    "In-flight Service": 5,
    "In-flight Wifi Service": 2,
    "In-flight Entertainment": 4,
    "Baggage Handling": 5,
}


def health(timeout: float = 5.0) -> Dict[str, Any]:
    """Return the API health payload, or a fallback dict if unreachable."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=timeout)
        return response.json()
    except requests.exceptions.RequestException:
        return {"status": "unreachable", "model_loaded": False}


def predict(passenger: Dict[str, Any], timeout: float = 10.0) -> requests.Response:
    """POST a passenger dict to /predict and return the raw Response.

    Raises:
        requests.exceptions.ConnectionError: if the API is unreachable.
    """
    return requests.post(f"{API_URL}/predict", json=passenger, timeout=timeout)
