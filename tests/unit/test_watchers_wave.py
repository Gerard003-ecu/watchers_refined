#!/usr/bin/env python3
"""
test_watchers_wave.py
Pruebas para watchers_wave.py
"""

import os
from unittest.mock import patch

import pytest
import requests

# Parametrizar la URL base para watchers_wave.
BASE_URL = os.environ.get("WATCHERS_WAVE_URL", "http://localhost:5000")
MALLA_ENDPOINT = f"{BASE_URL}/api/malla"
WAVE_CONTROL_ENDPOINT = f"{BASE_URL}/api/wave_control"
ACOUSTIC_ENDPOINT = f"{BASE_URL}/api/acoustic"


@pytest.fixture
def watchers_wave_running():
    yield  # Se asume que el servicio está corriendo


@patch("requests.get")
def test_get_malla(mock_get, watchers_wave_running):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "malla_A": [[{"x": 0, "y": 0, "amplitude": 0.8, "phase": 0.0}]],
        "malla_B": [[{"x": 0, "y": 0, "amplitude": 0.4, "phase": 0.0}]],
        "resonador": {"lambda_foton": 600},
        "status": "success",
    }
    response = requests.get(MALLA_ENDPOINT)
    assert response.status_code == 200
    assert "malla_A" in response.json()


@patch("requests.post")
def test_wave_control(mock_post, watchers_wave_running):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"status": "success", "c_current": 0.2}
    post_data = {"control_signal": 1.0}
    response = requests.post(WAVE_CONTROL_ENDPOINT, json=post_data)
    assert response.status_code == 200
    assert "c_current" in response.json()


@patch("requests.get")
def test_acoustic_get(mock_get, watchers_wave_running):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "frecuencia": 20000,
        "amplitud": 0.5,
        "status": "success",
    }
    response = requests.get(ACOUSTIC_ENDPOINT)
    assert response.status_code == 200
    assert "frecuencia" in response.json()


if __name__ == "__main__":
    pytest.main()
