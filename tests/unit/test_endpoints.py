# --- START OF FILE tests/unit/test_endpoints.py ---

# !/usr/bin/env python3
"""
test_endpoints.py - Pruebas para la API REST de AgentAI (Estratégico)
"""

import pytest
import threading
from unittest.mock import patch, MagicMock

# Importar la app Flask desde el módulo de endpoints refactorizado
# Ajusta la ruta si tu estructura es diferente
try:
    from agent_ai.api.endpoints import app as agent_ai_app
    # Importar la RUTA a la instancia para poder mockearla
    AGENT_AI_INSTANCE_PATH = 'agent_ai.api.endpoints.agent_ai_instance_app'
except ImportError:
    # Fallback por si la estructura es plana
    from endpoints import app as agent_ai_app
    AGENT_AI_INSTANCE_PATH = 'agent_ai.api.endpoints.agent_ai_instance_app'


@pytest.fixture
def client_and_mock():  # Renombrar para claridad
    """
    Configura un cliente de prueba
    Flask y la instancia mockeada de AgentAI.
    """
    agent_ai_app.config['TESTING'] = True
    agent_ai_app.config['WTF_CSRF_ENABLED'] = False
    with agent_ai_app.test_client() as client:
        # Mockear la instancia global donde está definida
        with patch(AGENT_AI_INSTANCE_PATH, autospec=True) as mock_instance:
            # Configurar atributos necesarios para health check por defecto
            # Esto evita errores si los tests no los configuran específicamente
            mock_instance._strategic_thread = MagicMock(
                spec=threading.Thread
            )  # Crear un mock del hilo
            mock_instance._strategic_thread.is_alive.return_value = True
            # Por defecto, decir que está vivo

            yield client, mock_instance


# --- Tests para Endpoints ---

def test_get_status_success(client_and_mock):
    """Prueba GET /api/status exitoso."""
    client, mock_instance = client_and_mock  # Desempaquetar

    mock_data = {
        "target_setpoint_vector": [1.0],
        "current_strategy": "test",
        "registered_modules": []
    }
    # Configurar el método del mock principal
    mock_instance.obtener_estado_completo.return_value = mock_data

    response = client.get("/api/status")

    # Verificar la llamada en el mock principal
    mock_instance.obtener_estado_completo.assert_called_once()
    assert response.status_code == 200
    assert response.json['status'] == 'success'
    assert response.json['data'] == mock_data


def test_post_strategic_command_success(client_and_mock):
    """Prueba POST /api/command con un comando estratégico válido."""
    client, mock_instance = client_and_mock
    payload = {"comando": "set_strategy", "valor": "performance"}
    mock_response = {"status": "success", "mensaje": "Estrategia establecida"}
    # Configurar el método del mock principal
    mock_instance.actualizar_comando_estrategico.return_value = mock_response

    response = client.post("/api/command", json=payload)

    # Verificar la llamada en el mock principal
    mock_instance.actualizar_comando_estrategico.assert_called_once_with(
        "set_strategy", "performance"
    )
    assert response.status_code == 200
    assert response.json == mock_response


def test_post_strategic_command_fail(client_and_mock):
    """Prueba POST /api/command con comando estratégico que falla."""
    client, mock_instance = client_and_mock
    payload = {"comando": "unknown_command", "valor": None}
    mock_response = {"status": "error", "mensaje": "Comando no reconocido"}
    # Configurar el método del mock principal
    mock_instance.actualizar_comando_estrategico.return_value = mock_response

    response = client.post("/api/command", json=payload)

    # Verificar la llamada en el mock principal
    mock_instance.actualizar_comando_estrategico.assert_called_once_with(
        "unknown_command", None
    )
    # El endpoint devuelve 400 si el comando falla
    assert response.status_code == 400
    assert response.json == mock_response


def test_post_command_missing_field(client_and_mock):
    """Prueba POST /api/command sin el campo 'comando'."""
    client, mock_instance = client_and_mock
    # No necesitamos mockear aquí, la validación está en el endpoint
    response = client.post("/api/command", json={"valor": "test"})
    assert response.status_code == 400
    assert "Falta el campo 'comando'" in response.json["mensaje"]


def test_register_module_success(client_and_mock):
    """Prueba POST /api/register exitoso."""
    client, mock_instance = client_and_mock
    module_data = {"nombre": "NewTool", "url": "http://newtool/health"}
    mock_response = {
        "status": "success",
        "mensaje": "Módulo 'NewTool' registrado"
    }
    # Configurar el método del mock principal
    mock_instance.registrar_modulo.return_value = mock_response

    response = client.post("/api/register", json=module_data)

    # Verificar la llamada en el mock principal
    mock_instance.registrar_modulo.assert_called_once_with(module_data)
    assert response.status_code == 200
    assert response.json == mock_response
    # IMPORTANTE: Al mockear registrar_modulo, evitamos que se lance
    # el hilo _validar_salud_modulo real


def test_register_module_fail(client_and_mock):
    """
    Prueba POST /api/register con
    fallo en la lógica interna.
    """
    client, mock_instance = client_and_mock
    module_data = {"nombre": "BadTool", "url": ""}  # URL inválida
    mock_response = {"status": "error",
    "mensaje": "Faltan campos obligatorios"
    }
    # Configurar el método del mock principal
    mock_instance.registrar_modulo.return_value = mock_response

    response = client.post("/api/register", json=module_data)

    # Verificar la llamada en el mock principal
    mock_instance.registrar_modulo.assert_called_once_with(module_data)
    # Endpoint devuelve 400 si el registro falla
    assert response.status_code == 400
    assert response.json == mock_response


def test_health_check_success(client_and_mock):
    """
    Prueba GET /api/health
    cuando el bucle está activo.
    """
    client, mock_instance = client_and_mock
    # Configurar el mock principal (ya debería tener _strategic_thread)
    mock_instance._strategic_thread.is_alive.return_value = True

    response = client.get("/api/health")

    # Verificar que se consultó el atributo
    mock_instance._strategic_thread.is_alive.assert_called_once()
    assert response.status_code == 200
    data = response.json
    assert data["status"] == "success"
    assert data["strategic_loop_active"] is True


def test_health_check_fail(client_and_mock):
    """
    Prueba GET /api/health
    cuando el bucle NO está activo.
    """
    client, mock_instance = client_and_mock
    # Configurar el mock principal
    mock_instance._strategic_thread.is_alive.return_value = False

    response = client.get("/api/health")

    # Verificar que se consultó el atributo
    mock_instance._strategic_thread.is_alive.assert_called_once()
    assert response.status_code == 503
    data = response.json
    assert data["status"] == "error"
    assert data["strategic_loop_active"] is False


def test_control_input_success(client_and_mock):
    """
    Prueba POST /api/control
    (entrada de cogniboard).
    """
    client, mock_instance = client_and_mock
    payload = {"control_signal": 0.85}
    # Configurar el método del mock principal
    # (no necesita return_value si no devuelve nada)

    response = client.post("/api/control", json=payload)

    # Verificar la llamada en el mock principal
    mock_instance.recibir_control_cogniboard.assert_called_once_with(0.85)
    assert response.status_code == 200
    assert response.json["status"] == "success"


def test_control_input_missing_data(client_and_mock):
    """
    Prueba POST /api/control
    sin 'control_signal'.
    """
    client, mock_instance = client_and_mock
    response = client.post("/api/control", json={})
    assert response.status_code == 400
    assert "Falta 'control_signal'" in response.json["mensaje"]


def test_config_input_success(client_and_mock):
    """
    Prueba POST /api/config
    (entrada de config_agent).
    """
    client, mock_instance = client_and_mock
    payload = {"config_status": {"docker": "ok", "network": "warning"}}
    # Configurar el método del mock principal
    # (no necesita return_value si no devuelve nada)

    response = client.post("/api/config", json=payload)

    # Verificar la llamada en el mock principal
    mock_instance.recibir_config_status.assert_called_once_with(
        {"docker": "ok", "network": "warning"}
    )
    assert response.status_code == 200
    assert response.json["status"] == "success"


def test_config_input_missing_data(client_and_mock):
    """
    Prueba POST /api/config
    sin 'config_status'.
    """
    client, mock_instance = client_and_mock
    response = client.post("/api/config", json={})
    assert response.status_code == 400
    assert "Falta 'config_status'" in response.json["mensaje"]

# --- END OF FILE tests/unit/test_endpoints.py ---
