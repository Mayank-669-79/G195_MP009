from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as app_main
from app.main import app


@pytest.fixture(autouse=True)
def mock_model():
    """Ensure a working mock model is present during API tests without needing files on disk."""
    original_model = app_main.model
    mock = MagicMock()
    mock.predict.return_value = np.array([32500.50])
    app_main.model = mock
    yield
    app_main.model = original_model


client = TestClient(app)


def test_health_endpoint():
    """GET /health must return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid():
    """POST /predict with valid fields returns status 200 and a numerical prediction."""
    payload = {
        "flats_occupied": 150,
        "temp_c": 32.5,
        "garden_area_m2": 850.0,
        "tank_level_morning_pct": 65.0,
        "leak_complaints": 3,
        "supply_hours": 6.5,
        "supply_source": "municipal",
        "tower": "Tower A"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_reject_temp_above_70():
    """Rule 1: temp_c above 70 is impossible and must be rejected with 422."""
    payload = {
        "flats_occupied": 150,
        "temp_c": 75.0,  # Invalid: > 70
        "garden_area_m2": 850.0,
        "tank_level_morning_pct": 65.0,
        "leak_complaints": 3,
        "supply_hours": 6.5,
        "supply_source": "municipal",
        "tower": "Tower A"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_reject_garden_below_0():
    """Rule 2: garden_area_m2 below 0 is always a typing mistake and must be rejected with 422."""
    payload = {
        "flats_occupied": 150,
        "temp_c": 32.5,
        "garden_area_m2": -15.0,  # Invalid: < 0
        "tank_level_morning_pct": 65.0,
        "leak_complaints": 3,
        "supply_hours": 6.5,
        "supply_source": "municipal",
        "tower": "Tower A"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_tower_category():
    """Unknown tower value must be rejected with 422."""
    payload = {
        "flats_occupied": 150,
        "temp_c": 32.5,
        "garden_area_m2": 500.0,
        "tank_level_morning_pct": 65.0,
        "leak_complaints": 3,
        "supply_hours": 6.5,
        "supply_source": "municipal",
        "tower": "Tower Unknown"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
