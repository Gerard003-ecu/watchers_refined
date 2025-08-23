# --- START OF FILE tests/unit/test_endpoints.py ---

# !/usr/bin/env python3
"""
test_endpoints.py - Pruebas para la API REST de AgentAI (Estratégico)
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

# Importar la app Flask desde el módulo de endpoints refactorizado
# Ajusta la ruta si tu estructura es diferente
from agent_ai.api.endpoints import app as agent_ai_app

# La ruta a mockear es ahora el módulo 'agent_ai_core' que se importa
# en 'endpoints.py'. Al mockear el módulo, podemos controlar su contenido,
# incluyendo la instancia 'agent_ai_instance'.
AGENT_AI_CORE_PATH = "agent_ai.api.endpoints.agent_ai_core"


@pytest.fixture
def client_and_mocks():
    """
    Configura un cliente de prueba Flask y mockea el módulo `agent_ai_core`
    entero, incluyendo su instancia y sus hilos.
    Devuelve el cliente, el mock de la instancia y el mock del módulo.
    """
    agent_ai_app.config["TESTING"] = True
    agent_ai_app.config["WTF_CSRF_ENABLED"] = False
    with agent_ai_app.test_client() as client:
        with patch(AGENT_AI_CORE_PATH) as mock_agent_ai_core:
            mock_instance = MagicMock()
            mock_agent_ai_core.agent_ai_instance = mock_instance
            mock_agent_ai_core._strategic_thread = MagicMock(spec=threading.Thread)
            mock_agent_ai_core.start_loop = MagicMock()
            mock_agent_ai_core.shutdown = MagicMock()
            yield client, mock_instance, mock_agent_ai_core


# --- Tests para Endpoints ---


def test_get_status_success(client_and_mocks):
    """Prueba GET /api/status exitoso."""
    client, mock_instance, _ = client_and_mocks

    mock_data = {
        "target_setpoint_vector": [1.0],
        "current_strategy": "test",
        "registered_modules": [],
    }
    mock_instance.obtener_estado_completo.return_value = mock_data

    response = client.get("/api/status")

    mock_instance.obtener_estado_completo.assert_called_once()
    assert response.status_code == 200
    assert response.json["status"] == "success"
    assert response.json["data"] == mock_data


def test_post_strategic_command_success(client_and_mocks):
    """Prueba POST /api/command con un comando estratégico válido."""
    client, mock_instance, _ = client_and_mocks
    payload = {"comando": "set_strategy", "valor": "performance"}
    mock_response = {"status": "success", "mensaje": "Estrategia establecida"}
    mock_instance.actualizar_comando_estrategico.return_value = mock_response

    response = client.post("/api/command", json=payload)

    mock_instance.actualizar_comando_estrategico.assert_called_once_with(
        "set_strategy", "performance"
    )
    assert response.status_code == 200
    assert response.json == mock_response


def test_post_strategic_command_fail(client_and_mocks):
    """Prueba POST /api/command con comando estratégico que falla."""
    client, mock_instance, _ = client_and_mocks
    payload = {"comando": "unknown_command", "valor": None}
    mock_response = {"status": "error", "mensaje": "Comando no reconocido"}
    mock_instance.actualizar_comando_estrategico.return_value = mock_response

    response = client.post("/api/command", json=payload)

    mock_instance.actualizar_comando_estrategico.assert_called_once_with(
        "unknown_command", None
    )
    assert response.status_code == 400
    assert response.json == mock_response


def test_post_command_missing_field(client_and_mocks):
    """Prueba POST /api/command sin el campo 'comando'."""
    client, _, _ = client_and_mocks
    response = client.post("/api/command", json={"valor": "test"})
    assert response.status_code == 400
    assert "Falta el campo 'comando'" in response.json["mensaje"]


def test_register_module_success(client_and_mocks):
    """Prueba POST /api/register exitoso."""
    client, mock_instance, _ = client_and_mocks
    module_data = {"nombre": "NewTool", "url": "http://newtool/health"}
    mock_response = {"status": "success", "mensaje": "Módulo 'NewTool' registrado"}
    mock_instance.registrar_modulo.return_value = mock_response

    response = client.post("/api/register", json=module_data)

    mock_instance.registrar_modulo.assert_called_once_with(module_data)
    assert response.status_code == 200
    assert response.json == mock_response


def test_register_module_fail(client_and_mocks):
    """Prueba POST /api/register con fallo en la lógica interna."""
    client, mock_instance, _ = client_and_mocks
    module_data = {"nombre": "BadTool", "url": ""}
    mock_response = {
        "status": "error",
        "mensaje": "Faltan campos obligatorios",
    }
    mock_instance.registrar_modulo.return_value = mock_response

    response = client.post("/api/register", json=module_data)

    mock_instance.registrar_modulo.assert_called_once_with(module_data)
    assert response.status_code == 400
    assert response.json == mock_response


def test_health_check_success(client_and_mocks):
    """Prueba GET /api/health cuando el bucle está activo."""
    client, _, mock_agent_ai_core = client_and_mocks
    mock_agent_ai_core._strategic_thread.is_alive.return_value = True

    response = client.get("/api/health")

    mock_agent_ai_core._strategic_thread.is_alive.assert_called_once()
    assert response.status_code == 200
    data = response.json
    assert data["status"] == "success"
    assert data["strategic_loop_active"] is True


def test_health_check_fail(client_and_mocks):
    """Prueba GET /api/health cuando el bucle NO está activo."""
    client, _, mock_agent_ai_core = client_and_mocks
    mock_agent_ai_core._strategic_thread.is_alive.return_value = False

    response = client.get("/api/health")

    mock_agent_ai_core._strategic_thread.is_alive.assert_called_once()
    assert response.status_code == 503
    data = response.json
    assert data["status"] == "error"
    assert data["strategic_loop_active"] is False


def test_control_input_success(client_and_mocks):
    """Prueba POST /api/control (entrada de cogniboard)."""
    client, mock_instance, _ = client_and_mocks
    payload = {"control_signal": 0.85}

    response = client.post("/api/control", json=payload)

    mock_instance.recibir_control_cogniboard.assert_called_once_with(0.85)
    assert response.status_code == 200
    assert response.json["status"] == "success"


def test_control_input_missing_data(client_and_mocks):
    """Prueba POST /api/control sin 'control_signal'."""
    client, _, _ = client_and_mocks
    response = client.post("/api/control", json={})
    assert response.status_code == 400
    assert "Falta 'control_signal'" in response.json["mensaje"]


def test_config_input_success(client_and_mocks):
    """Prueba POST /api/config (entrada de config_agent)."""
    client, mock_instance, _ = client_and_mocks
    payload = {"config_status": {"docker": "ok", "network": "warning"}}

    response = client.post("/api/config", json=payload)

    mock_instance.recibir_config_status.assert_called_once_with(
        {"docker": "ok", "network": "warning"}
    )
    assert response.status_code == 200
    assert response.json["status"] == "success"


def test_config_input_missing_data(client_and_mocks):
    """Prueba POST /api/config sin 'config_status'."""
    client, _, _ = client_and_mocks
    response = client.post("/api/config", json={})
    assert response.status_code == 400
    assert "Falta 'config_status'" in response.json["mensaje"]


def test_receive_metrics_success(client_and_mocks):
    """Prueba POST /api/metrics exitoso."""
    client, mock_instance, _ = client_and_mocks

    metric_data = {
        "source_service": "test_service",
        "function_name": "test_function",
        "execution_time": 0.123,
        "call_count": 5,
    }

    response = client.post("/api/metrics", json=metric_data)

    mock_instance.store_metric.assert_called_once_with(metric_data)
    assert response.status_code == 200
    assert response.json["status"] == "success"


def test_receive_metrics_bad_request(client_and_mocks):
    """Prueba POST /api/metrics con payload vacío."""
    client, mock_instance, _ = client_and_mocks

    # Enviar un payload vacío/nulo
    response = client.post("/api/metrics", json=None)

    mock_instance.store_metric.assert_not_called()
    assert response.status_code == 400
    assert "Payload JSON vacío o ausente" in response.json["mensaje"]


def test_receive_metrics_success(client_and_mocks):
    """Prueba POST /api/metrics exitoso."""
    client, mock_instance, _ = client_and_mocks

    metric_data = {
        "source_service": "test_service",
        "function_name": "test_function",
        "execution_time": 0.123,
        "call_count": 5,
    }

    response = client.post("/api/metrics", json=metric_data)

    mock_instance.store_metric.assert_called_once_with(metric_data)
    assert response.status_code == 200
    assert response.json["status"] == "success"


def test_receive_metrics_bad_request(client_and_mocks):
    """Prueba POST /api/metrics con payload vacío."""
    client, mock_instance, _ = client_and_mocks

    response = client.post("/api/metrics", json=None)

    mock_instance.store_metric.assert_not_called()
    assert response.status_code == 400
    assert "Payload JSON vacío o ausente" in response.json["mensaje"]


# --- END OF FILE tests/unit/test_endpoints.py ---
