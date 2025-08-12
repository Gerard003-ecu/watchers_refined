# tests/integration/test_atomic_piston_api.py
import pytest
import json
# tests/integration/test_atomic_piston_api.py
import pytest
import json
import threading
# Se importa el módulo directamente para poder modificar sus variables globales en el entorno de prueba
from atomic_piston import atomic_piston_service
from atomic_piston.atomic_piston_service import app as agent_api, AtomicPiston
from atomic_piston.constants import PistonMode
from atomic_piston.config import PistonConfig

# 1. Configuración y Fixtures (pytest)

@pytest.fixture
def client():
    """
    Fixture de cliente de prueba de Flask.

    Crea un cliente de prueba para la API de Flask que se puede usar en los tests
    para realizar solicitudes HTTP sin necesidad de un servidor real.
    """
    with agent_api.test_client() as client:
        yield client

@pytest.fixture(autouse=True)
def reset_ipu_instance():
    """
    Fixture para reiniciar la instancia de IPU y la configuración antes de cada test.

    Esto asegura que cada test se ejecute en un estado limpio y conocido,
    previniendo que los resultados de un test afecten a otro.
    """
    # Se inicializa el objeto de configuración que el servicio espera
    atomic_piston_service.config = PistonConfig()

    # Se modifica la instancia directamente en el módulo del servicio para que los endpoints la vean.
    atomic_piston_service.ipu_instance = AtomicPiston(
        capacity=10.0,
        elasticity=100.0,
        damping=5.0,
        piston_mass=1.0,
        # Usar el modelo de fricción de la configuración por defecto para consistencia
        friction_model=atomic_piston_service.config.friction_model
    )
    # Se usa un mock de hilo no vivo para que `is_alive()` devuelva False, haciendo que
    # el detalle 'simulation_running' en /api/health sea un booleano False en lugar de None.
    atomic_piston_service.simulation_thread = threading.Thread()

# 2. Suite de Pruebas para cada Endpoint

class TestHealthEndpoint:
    """Suite de pruebas para el endpoint /api/health."""

    def test_health_reports_error_when_simulation_is_down(self, client):
        """
        Verifica que /api/health devuelve 503 si la simulación no está activa,
        incluso con el pistón inicializado, como es el caso en el entorno de test.
        """
        # WHEN
        response = client.get('/api/health')

        # THEN
        assert response.status_code == 503
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'El hilo de simulación está inactivo' in data['message']
        assert data['details']['simulation_running'] is False
        assert data['details']['piston_initialized'] is True


class TestStateEndpoint:
    """Suite de pruebas para el endpoint /api/state."""

    def test_get_state_success(self, client):
        """
        Verifica que se puede obtener el estado del pistón y que tiene la estructura correcta.
        """
        # WHEN
        response = client.get('/api/state')

        # THEN
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'state' in data
        state = data['state']
        expected_keys = [
            "position", "velocity", "acceleration", "mode", "stored_energy",
            "current_charge", "circuit_voltage", "circuit_current", "output_voltage",
            "battery_mode_status", "control_targets"
        ]
        for key in expected_keys:
            assert key in state


class TestConfigEndpoint:
    """Suite de pruebas para el endpoint /api/config."""

    def test_get_config_success(self, client):
        """
        Verifica que se puede obtener la configuración estática del pistón.
        """
        # WHEN
        response = client.get('/api/config')

        # THEN
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'config' in data
        config = data['config']
        expected_keys = [
            "physical_params", "transducer_params", "friction_params",
            "electrical_equivalent_params", "operational_params", "pid_gains"
        ]
        for key in expected_keys:
            assert key in config


class TestControlEndpoint:
    """Suite de pruebas para el endpoint /api/control."""

    def test_set_control_signal_success(self, client):
        """
        Verifica que una señal de control válida actualiza el 'target_energy'.
        """
        # GIVEN
        payload = {"control_signal": 50.5}

        # WHEN
        response = client.post('/api/control', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['new_energy_target'] == 50.5

        # Confirmar el cambio de estado
        state_response = client.get('/api/state')
        state_data = json.loads(state_response.data)
        assert state_data['state']['control_targets']['target_energy'] == 50.5

    def test_set_control_signal_invalid_payload(self, client):
        """Verifica que un payload sin 'control_signal' devuelve 400."""
        # GIVEN
        payload = {"invalid_key": 123}

        # WHEN
        response = client.post('/api/control', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert "falta 'control_signal'" in data['message']

    def test_set_control_signal_non_numeric(self, client):
        """Verifica que un 'control_signal' no numérico devuelve 400."""
        # GIVEN
        payload = {"control_signal": "not-a-number"}

        # WHEN
        response = client.post('/api/control', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert "'control_signal' debe ser un valor numérico" in data['message']

    def test_set_control_signal_malformed_json(self, client):
        """Verifica que un JSON malformado devuelve 400."""
        # GIVEN
        malformed_json = '{"control_signal": 123.4' # JSON inválido

        # WHEN
        response = client.post('/api/control', data=malformed_json, content_type='application/json')

        # THEN
        assert response.status_code == 400 # Flask maneja esto automáticamente


class TestCommandEndpoint:
    """Suite de pruebas para el endpoint /api/command."""

    def test_command_set_mode_to_battery_succeeds(self, client):
        """Verifica que el comando 'set_mode' a 'battery' funciona."""
        # GIVEN
        payload = {"command": "set_mode", "value": "battery"}

        # WHEN
        response = client.post('/api/command', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['message'] == "Modo cambiado a battery."

        # Confirmar el cambio de estado
        state_response = client.get('/api/state')
        state_data = json.loads(state_response.data)
        assert state_data['state']['mode'] == 'battery'

    def test_command_trigger_discharge_succeeds(self, client):
        """Verifica que 'trigger_discharge' funciona en modo batería."""
        # GIVEN: Primero, cambiar a modo batería
        client.post('/api/command', data=json.dumps({"command": "set_mode", "value": "battery"}), content_type='application/json')

        payload = {"command": "trigger_discharge", "value": True}

        # WHEN
        response = client.post('/api/command', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'

        # Confirmar el cambio de estado
        state_response = client.get('/api/state')
        state_data = json.loads(state_response.data)
        assert state_data['state']['battery_mode_status']['is_discharging'] is True

    def test_command_reset_succeeds(self, client):
        """Verifica que el comando 'reset' reinicia el estado del pistón."""
        # GIVEN: Cambiar algún estado directamente en la instancia del servicio
        atomic_piston_service.ipu_instance.position = 5.0
        atomic_piston_service.ipu_instance.velocity = 2.0

        payload = {"command": "reset"}

        # WHEN
        response = client.post('/api/command', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert "reiniciado" in data['message']

        # Confirmar el estado reseteado
        state_response = client.get('/api/state')
        state_data = json.loads(state_response.data)
        assert state_data['state']['position'] == 0.0
        assert state_data['state']['velocity'] == 0.0

    def test_command_unknown_command_fails(self, client):
        """Verifica que un comando desconocido devuelve un error 400."""
        # GIVEN
        payload = {"command": "non_existent_command", "value": 123}

        # WHEN
        response = client.post('/api/command', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert "Comando desconocido" in data['message']

    def test_command_set_mode_invalid_value_fails(self, client):
        """Verifica que un valor inválido para 'set_mode' devuelve 400."""
        # GIVEN
        payload = {"command": "set_mode", "value": "invalid_mode"}

        # WHEN
        response = client.post('/api/command', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert "Valor inválido" in data['message']

    def test_command_set_energy_target_non_numeric_fails(self, client):
        """Verifica que un valor no numérico para 'set_energy_target' devuelve 400."""
        # GIVEN
        payload = {"command": "set_energy_target", "value": "not-a-float"}

        # WHEN
        response = client.post('/api/command', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert "Valor inválido" in data['message']

    def test_command_payload_missing_command_fails(self, client):
        """Verifica que un payload sin la clave 'command' devuelve 400."""
        # GIVEN
        payload = {"value": 123} # Falta la clave 'command'

        # WHEN
        response = client.post('/api/command', data=json.dumps(payload), content_type='application/json')

        # THEN
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert "falta 'command'" in data['message']
