"""
Pruebas integrales para el dashboard de Watchers.
"""

import pytest
from unittest.mock import patch, MagicMock
from dashboard import (
    obtener_datos_reales,
    obtener_estado_malla_sim,
    crear_grafico_barras,
    crear_control,
    app
)


# Fixture para cliente de prueba
@pytest.fixture
def dash_client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# Pruebas de componentes básicos
def test_obtener_estado_malla_sim():
    data = obtener_estado_malla_sim()
    assert "malla_A" in data
    assert len(data["malla_A"][0]) == 1
    assert data["resonador"]["lambda_foton"] == 600


def test_crear_grafico_barras():
    fig = crear_grafico_barras({"A": 0.8, "B": 0.4}, "ambas")
    assert len(fig.data) == 2
    assert fig.layout.title.text == "Amplitud Promedio por Malla"


# Pruebas de integración con mocks
@patch('dashboard.requests.get')
def test_obtener_datos_reales_success(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "success"}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp
    
    result = obtener_datos_reales("http://fake.api/status")
    assert "status" in result
    assert result["status"] == "success"


@patch('dashboard.requests.get')
def test_obtener_datos_reales_error(mock_get):
    mock_get.side_effect = Exception("Error de conexión")
    result = obtener_datos_reales("http://fake.api/status")
    assert "error" in result
    assert "Error de red" in result["error"]


# Pruebas de endpoints del dashboard
def test_dash_layout(dash_client):
    response = dash_client.get("/")
    assert response.status_code == 200
    assert "Panel de Control Watchers" in str(response.data)


def test_dash_interaction(dash_client):
    response = dash_client.post("/_dash-update-component", json={})
    assert response.status_code == 200


# Pruebas de generación de controles
def test_crear_control_button():
    control = crear_control({"ui_type": "button", "config": {"label": "Test"}})
    assert "Test" in str(control)


def test_crear_control_slider():
    control = crear_control(
        {"ui_type": "slider", "config": {"min": 0, "max": 100}}
    )
    assert "Slider" in str(control)


if __name__ == "__main__":
    pytest.main()
