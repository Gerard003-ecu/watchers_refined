#!/usr/bin/env python3
"""
test_watcher_focus.py
"""

import pytest
import requests
from unittest.mock import patch
# Import current_state directly as it's used elsewhere and seems fine
from watchers.watchers_tools.watcher_focus.watcher_focus import (
    current_state
    simulate_watcher_focus_infinite
)
import threading
import time
import os

# Parametrizar la URL base para el módulo watcher_focus.
FOCUS_URL = os.environ.get("WATCHER_FOCUS_URL", "http://localhost:6000")


@pytest.fixture
def watcher_focus_thread_manager():
    """Inicia y detiene el hilo de simulación para pruebas."""
    # Access simulate_watcher_focus_infinite function
    # from the watcher_focus module.
    # Ahora puedes usar la función importada directamente
    thread = threading.Thread(
        target=simulate_watcher_focus_infinite, # Usar la función directamente
        args=(0.001,),
        daemon=True
    )
    thread.start()
    
    # Dar tiempo al hilo para que se inicie y posiblemente ejecute un ciclo
    time.sleep(0.01) 
    
    # 'yield' pasa el control al test que usa esta fixture.
    # El código después de yield se ejecutará cuando el test termine.
    yield 
    # El hilo daemon se detendrá al finalizar el test


def test_simulate_watcher_focus(watcher_focus_thread):
    """Verifica que el estado se actualice después de iniciar la simulación."""
    state = current_state
    assert state["x"] is not None and \
           state["y"] is not None, "El estado no se actualizó correctamente."


@patch("requests.get")
def test_get_focus_endpoint(mock_get):
    """Prueba el endpoint /api/focus simulando una respuesta."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "status": "success",
        "focus_state": {
            "t": 0.1,
            "x": 1.0,
            "y": 0.0,
            "z": 0.5,
            "phase": 0.0,
            "z_error": 0.5
        }
    }
    response = requests.get(FOCUS_URL + "/api/focus")
    mock_get.assert_called_once()
    assert response.status_code == 200, \
        "El endpoint /api/focus no respondió correctamente."
