# --- START OF FILE tests/unit/test_harmony_controller.py (AJUSTADO Fase 1) ---

import unittest
import unittest.mock as mock
import json
import numpy as np
import os
import requests

# --- Configuración de Entorno Simulada ANTES de importar ---
if 'WATCHERS_TOOLS_CONFIG' in os.environ:
    del os.environ['WATCHERS_TOOLS_CONFIG']
os.environ['ECU_API_URL'] = "http://fake-ecu:8000/api/ecu"
os.environ['HC_SETPOINT_VECTOR'] = '[1.0, 0.0]'

# --- Importar Módulo Bajo Prueba ---
# Usar un bloque try/except para manejar posible error
try:
    from control import harmony_controller
    from control.harmony_controller import (
        HarmonyControllerState,
        get_ecu_state,
        get_tool_state,
        send_tool_control,
    )
except ImportError as e:
    raise ImportError(
        "No se pudo importar harmony_controller. "
        "Verifica la estructura del proyecto y PYTHONPATH. "
        f"Error: {e}"
    )

# --- Mock Global de Requests ---
mock_requests = mock.MagicMock()


class MockResponse:
    def __init__(self, json_data, status_code, text=""):
        self._json_data = json_data
        self.status_code = status_code
        self.text = text if json_data is None else json.dumps(json_data)

    def json(self):
        if self._json_data is None:
            # Simular error de decodificación
            raise requests.exceptions.JSONDecodeError(
                "Mocked decode error", "", 0
            )
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"Mocked HTTP Error {self.status_code}", response=self
            )


# --- Tests para HarmonyControllerState (AJUSTADOS) ---
class TestHarmonyControllerState(unittest.TestCase):
    def setUp(self):
        self.test_kp = harmony_controller.KP_INIT
        self.test_ki = harmony_controller.KI_INIT
        self.test_kd = harmony_controller.KD_INIT
        self.test_setpoint_vector = (
            harmony_controller.setpoint_vector_init.tolist()
        )
        self.test_setpoint_value = harmony_controller.setpoint_init
        self.state = HarmonyControllerState(
            kp=self.test_kp,
            ki=self.test_ki,
            kd=self.test_kd,
            initial_setpoint=self.test_setpoint_value,
            initial_setpoint_vector=self.test_setpoint_vector
        )

    def test_initialization(self):
        """Verifica estado inicial (sin tools gestionados)."""
        self.assertEqual(self.state.pid_controller.Kp, self.test_kp)
        self.assertEqual(self.state.pid_controller.Ki, self.test_ki)
        self.assertEqual(self.state.pid_controller.Kd, self.test_kd)
        self.assertAlmostEqual(
            self.state.current_setpoint, self.test_setpoint_value
        )
        self.assertEqual(self.state.setpoint_vector, self.test_setpoint_vector)
        self.assertEqual(self.state.last_ecu_state, [])
        # Verificar que managed_tools_details inicia vacío
        self.assertEqual(self.state.managed_tools_details, {})

    def test_update_setpoint_value_only(self):
        """Prueba actualizar setpoint solo con valor."""
        new_val = 5.5
        initial_vec = list(self.state.setpoint_vector)
        self.state.update_setpoint(new_val)
        self.assertEqual(self.state.current_setpoint, new_val)
        self.assertEqual(self.state.pid_controller.setpoint, new_val)
        # Vector no debe cambiar
        self.assertEqual(self.state.setpoint_vector, initial_vec)

    def test_update_setpoint_with_vector(self):
        """Prueba actualizar setpoint con vector."""
        new_vec = [3.0, 4.0]
        new_val = np.linalg.norm(new_val)
        self.state.update_setpoint(new_val, new_vec)
        self.assertAlmostEqual(self.state.current_setpoint, 5.0)
        self.assertAlmostEqual(self.state.pid_controller.setpoint, 5.0)
        self.assertEqual(self.state.setpoint_vector, [3.0, 4.0])

    # Tests para registro/desregistro de tools
    def test_register_managed_tool_new(self):
        """Prueba registrar un nuevo tool."""
        self.state.register_managed_tool(
            "tool_a", "http://a", "malla_watcher", "potenciador"
        )
        self.assertIn("tool_a", self.state.managed_tools_details)
        details = self.state.managed_tools_details["tool_a"]
        self.assertEqual(details["url"], "http://a")
        self.assertEqual(details["aporta_a"], "malla_watcher")
        self.assertEqual(details["naturaleza"], "potenciador")
        self.assertEqual(details["last_state"], {"status": "unknown"})
        self.assertEqual(details["last_control"], 0.0)

    def test_register_managed_tool_update(self):
        """Prueba actualizar un tool existente."""
        self.state.register_managed_tool(
            "tool_a", "http://a", "malla_watcher", "potenciador"
        )
        # Registrar de nuevo con diferente URL o naturaleza
        self.state.register_managed_tool(
            "tool_a", "http://a_new", "malla_watcher", "reductor"
        )
        self.assertIn("tool_a", self.state.managed_tools_details)
        details = self.state.managed_tools_details["tool_a"]
        self.assertEqual(details["url"], "http://a_new")
        # Verifica actualización
        self.assertEqual(details["naturaleza"], "reductor")

    def test_unregister_managed_tool_existing(self):
        """Prueba eliminar un tool existente."""
        self.state.register_managed_tool(
            "tool_a", "http://a", "malla_watcher", "potenciador"
        )
        self.assertIn("tool_a", self.state.managed_tools_details)
        self.state.unregister_managed_tool("tool_a")
        self.assertNotIn("tool_a", self.state.managed_tools_details)

    def test_unregister_managed_tool_non_existing(self):
        """Prueba eliminar un tool que no existe."""
        self.assertNotIn("tool_b", self.state.managed_tools_details)
        # No debería lanzar excepción
        self.state.unregister_managed_tool("tool_b")
        self.assertNotIn("tool_b", self.state.managed_tools_details)

    # MODIFICADO: test_get_state_snapshot refleja nueva estructura
    def test_get_state_snapshot(self):
        """Prueba obtener snapshot del estado."""
        self.state.register_managed_tool(
            "tool_a", "http://a", "malla_watcher", "potenciador"
        )
        self.state.register_managed_tool(
            "tool_b", "http://b", "matriz_ecu", "reductor"
        )
        self.state.managed_tools_details["tool_a"]["last_state"] = {"amp": 1.5}
        self.state.managed_tools_details["tool_b"]["last_control"] = -0.2

        snapshot = self.state.get_state_snapshot()

        self.assertIsInstance(snapshot, dict)
        self.assertAlmostEqual(
            snapshot["setpoint_value"], self.test_setpoint_value
        )
        self.assertEqual(snapshot["setpoint_vector"], self.test_setpoint_vector)
        self.assertEqual(snapshot["last_measurement"], 0.0)
        self.assertEqual(snapshot["last_pid_output"], 0.0)
        self.assertEqual(snapshot["last_ecu_state"], [])
        self.assertEqual(snapshot["pid_gains"]["Kp"], self.test_kp)

        self.assertIn("managed_tools", snapshot)
        managed_tools_snap = snapshot["managed_tools"]
        self.assertIn("tool_a", managed_tools_snap)
        self.assertIn("tool_b", managed_tools_snap)
        self.assertEqual(managed_tools_snap["tool_a"]["url"], "http://a")
        self.assertEqual(
            managed_tools_snap["tool_a"]["aporta_a"], "malla_watcher"
        )
        self.assertEqual(
            managed_tools_snap["tool_a"]["naturaleza"], "potenciador"
        )
        self.assertEqual(
            managed_tools_snap["tool_a"]["last_state"], {"amp": 1.5}
        )
        self.assertEqual(
            managed_tools_snap["tool_b"]["aporta_a"], "matriz_ecu"
        )
        self.assertEqual(
            managed_tools_snap["tool_b"]["naturaleza"], "reductor"
        )
        self.assertEqual(
            managed_tools_snap["tool_b"]["last_control"], -0.2
        )


# --- Tests para Funciones de Comunicación ---
@mock.patch('control.harmony_controller.requests', mock_requests)
class TestCommunicationFunctions(unittest.TestCase):

    def setUp(self):
        mock_requests.reset_mock()
        mock_requests.get.side_effect = None
        mock_requests.post.side_effect = None

    def test_get_ecu_state_success(self):
        """Prueba obtener estado ECU exitoso."""
        mock_response_data = {
            "status": "success",
            "estado_campo_unificado": [[0.5, 0.1], [-0.2, 0.3]]
        }
        mock_requests.get.return_value = MockResponse(mock_response_data, 200)
        state = get_ecu_state()
        mock_requests.get.assert_called_once_with(
            harmony_controller.ECU_API_URL,
            timeout=harmony_controller.REQUESTS_TIMEOUT
        )
        self.assertEqual(state, [[0.5, 0.1], [-0.2, 0.3]])

    def test_get_ecu_state_network_error_with_retry(self):
        """Prueba error de red con reintentos."""
        mock_requests.get.side_effect = requests.exceptions.RequestException(
            "Network Error"
        )
        with mock.patch('time.sleep') as mock_sleep:
            state = get_ecu_state()
            self.assertIsNone(state)
            self.assertEqual(
                mock_requests.get.call_count,
                harmony_controller.MAX_RETRIES
            )
            self.assertEqual(
                mock_sleep.call_count,
                harmony_controller.MAX_RETRIES - 1
            )

    def test_get_ecu_state_bad_json(self):
        """Prueba JSON inválido."""
        mock_requests.get.return_value = MockResponse(
            None, 200, text="<html>bad</html>"
        )
        state = get_ecu_state()
        self.assertIsNone(state)

    def test_get_ecu_state_invalid_structure(self):
        """Prueba estructura inválida."""
        mock_response_data = {"status": "success", "message": "No state found"}
        mock_requests.get.return_value = MockResponse(mock_response_data, 200)
        state = get_ecu_state()
        self.assertIsNone(state)

    def test_get_tool_state_success(self):
        """Prueba obtener estado de tool exitoso."""
        mock_tool_name = "tool_x"
        mock_tool_url = "http://toolx:9999"
        expected_state_data = {"value": 42, "mode": "auto"}
        mock_response_data = {"status": "success", "state": expected_state_data}
        mock_requests.get.return_value = MockResponse(mock_response_data, 200)

        state = get_tool_state(mock_tool_name, mock_tool_url)

        mock_requests.get.assert_called_once_with(
            f"{mock_tool_url}/api/state",
            timeout=harmony_controller.REQUESTS_TIMEOUT
        )
        self.assertEqual(state, expected_state_data)

    def test_get_tool_state_error_with_retry(self):
        """Prueba error de red con reintentos."""
        timeout_exception = requests.exceptions.Timeout("Timeout Error")
        mock_requests.get.side_effect = timeout_exception
        with mock.patch('time.sleep') as mock_sleep:
            state = get_tool_state("tool_y", "http://tooly:8888")
            expected_error = {
                "status": "error",
                "message": (
                    f"No se pudo obtener estado después de "
                    f"{harmony_controller.MAX_RETRIES} intentos"
                )
            }
            self.assertEqual(state, expected_error)
            self.assertEqual(
                mock_requests.get.call_count,
                harmony_controller.MAX_RETRIES
            )
            self.assertEqual(
                mock_sleep.call_count,
                harmony_controller.MAX_RETRIES - 1
            )

    def test_send_tool_control_success(self):
        """Prueba enviar control a tool exitoso."""
        mock_requests.post.return_value = MockResponse({"status": "success"}, 200)
        success = send_tool_control("tool_z", "http://toolz:7777", 0.75)
        self.assertTrue(success)
        mock_requests.post.assert_called_once_with(
            "http://toolz:7777/api/control",
            json={"control_signal": 0.75},
            timeout=harmony_controller.REQUESTS_TIMEOUT
        )

    def test_send_tool_control_retry_success(self):
        """Prueba enviar control con reintentos exitosos."""
        mock_requests.post.side_effect = [
            requests.exceptions.RequestException("Attempt 1 fail"),
            requests.exceptions.RequestException("Attempt 2 fail"),
            MockResponse({"status": "success"}, 200)
        ]
        with mock.patch('time.sleep') as mock_sleep:
            success = send_tool_control("tool_w", "http://toolw:6666", -0.5)
            self.assertTrue(success)
            self.assertEqual(mock_requests.post.call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)
            mock_requests.post.assert_called_with(
                "http://toolw:6666/api/control",
                json={"control_signal": -0.5},
                timeout=harmony_controller.REQUESTS_TIMEOUT
            )

    def test_send_tool_control_retry_failure(self):
        """Prueba enviar control con reintentos fallidos."""
        mock_requests.post.side_effect = requests.exceptions.RequestException(
            "Persistent fail"
        )
        with mock.patch('time.sleep') as mock_sleep:
            success = send_tool_control("tool_v", "http://toolv:5555", 0.1)
            self.assertFalse(success)
            self.assertEqual(
                mock_requests.post.call_count,
                harmony_controller.MAX_RETRIES
            )
            self.assertEqual(
                mock_sleep.call_count,
                harmony_controller.MAX_RETRIES - 1
            )


# --- Tests para API Flask ---
@mock.patch('control.harmony_controller.requests', mock_requests)
class TestHarmonyControllerAPI(unittest.TestCase):

    def setUp(self):
        harmony_controller.app.config['TESTING'] = True
        self.client = harmony_controller.app.test_client()
        harmony_controller.controller_state = HarmonyControllerState()
        mock_requests.reset_mock()
        mock_requests.get.side_effect = None
        mock_requests.post.side_effect = None
        self.mock_pid_reset = mock.patch.object(
            harmony_controller.controller_state.pid_controller, 'reset'
        ).start()
        self.addCleanup(mock.patch.stopall)

    def test_get_harmony_state_api_initial(self):
        """Prueba obtener estado inicial."""
        response = self.client.get('/api/harmony/state')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['managed_tools'], {})

    # Tests para el endpoint de registro
    def test_register_tool_api_success(self):
        """Prueba registrar un tool vía API."""
        payload = {
            "nombre": "api_tool_1",
            "url": "http://apitool1",
            "aporta_a": "matriz_ecu",
            "naturaleza": "reductor"
        }
        response = self.client.post(
            '/api/harmony/register_tool', json=payload
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')

        self.assertIn(
            "api_tool_1",
            harmony_controller.controller_state.managed_tools_details
        )
        details = harmony_controller.controller_state.managed_tools_details[
            "api_tool_1"
        ]
        self.assertEqual(details["url"], "http://apitool1")
        self.assertEqual(details["aporta_a"], "matriz_ecu")
        self.assertEqual(details["naturaleza"], "reductor")

        response_state = self.client.get('/api/harmony/state')
        state_data = json.loads(response_state.data)['data']
        self.assertIn("api_tool_1", state_data['managed_tools'])
        self.assertEqual(
            state_data['managed_tools']['api_tool_1']['naturaleza'],
            "reductor"
        )

    def test_register_tool_api_missing_fields(self):
        """Prueba campos faltantes."""
        payload = {"nombre": "bad_tool", "url": "http://bad"}
        response = self.client.post(
            '/api/harmony/register_tool', json=payload
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertIn("ausente o inválido", data['message'])

    def test_register_tool_api_invalid_type(self):
        """Prueba tipo inválido."""
        payload = {
            "nombre": "bad_type",
            "url": 123,
            "aporta_a": "ecu",
            "naturaleza": "mod"
        }
        response = self.client.post(
            '/api/harmony/register_tool', json=payload
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertIn("ausente o inválido", data['message'])

    # --- Tests para otros endpoints ---
    def test_set_harmony_setpoint_value_api(self):
        """Prueba actualizar setpoint con valor."""
        payload = {"setpoint_value": 9.87}
        response = self.client.post('/api/harmony/setpoint', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['new_setpoint_value'], 9.87)
        self.assertEqual(
            harmony_controller.controller_state.current_setpoint, 9.87
        )

    def test_set_harmony_setpoint_vector_api(self):
        """Prueba actualizar setpoint con vector."""
        payload = {"setpoint_vector": [6.0, 8.0]}
        expected_norm = 10.0
        response = self.client.post('/api/harmony/setpoint', json=payload)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertAlmostEqual(data['new_setpoint_value'], expected_norm)
        self.assertEqual(data['new_setpoint_vector'], [6.0, 8.0])
        self.assertAlmostEqual(
            harmony_controller.controller_state.current_setpoint,
            expected_norm
        )

    def test_set_harmony_setpoint_missing_data_api(self):
        """Prueba payload vacío."""
        response = self.client.post('/api/harmony/setpoint', json={})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertIn("payload json vacío o ausente", data['message'].lower())

    def test_set_harmony_setpoint_missing_keys_api(self):
        """Prueba claves requeridas faltantes."""
        payload = {"other_key": "some_value"}
        response = self.client.post('/api/harmony/setpoint', json=payload)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'error')
        self.assertIn(
            "se requiere 'setpoint_value' o 'setpoint_vector'",
            data['message'].lower()
        )

    def test_set_harmony_setpoint_bad_value_api(self):
        """Prueba valor inválido."""
        payload = {"setpoint_value": "not a float"}
        response = self.client.post('/api/harmony/setpoint', json=payload)
        self.assertEqual(response.status_code, 400)

    def test_reset_pid_api(self):
        """Prueba reinicio de PID."""
        response = self.client.post('/api/harmony/pid/reset')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['message'], 'PID reiniciado.')
        self.mock_pid_reset.assert_called_once()


if __name__ == '__main__':
    unittest.main()

# --- END OF FILE tests/unit/test_harmony_controller.py (AJUSTADO Fase 1) ---