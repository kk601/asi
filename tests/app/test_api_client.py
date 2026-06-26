"""Unit tests for the Streamlit API client."""

from unittest.mock import MagicMock, patch

import requests

from lib import api_client


def test_predict_posts_to_predict_endpoint():
    with patch("lib.api_client.requests.post") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        api_client.predict({"Age": 30})
        args, kwargs = mock_post.call_args
        assert args[0].endswith("/predict")
        assert kwargs["json"] == {"Age": 30}


def test_health_returns_fallback_on_connection_error():
    with patch(
        "lib.api_client.requests.get",
        side_effect=requests.exceptions.ConnectionError,
    ):
        result = api_client.health()
        assert result["model_loaded"] is False
        assert result["status"] == "unreachable"


def test_default_passenger_has_22_features():
    assert len(api_client.DEFAULT_PASSENGER) == 22
