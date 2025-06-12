#!/usr/bin/env python3
# --- START OF FILE tests/unit/test_agent_ai.py (AJUSTADO Fase 1) ---
"""
test_agent_ai.py - Pruebas unitarias para el núcleo estratégico 
AgentAI
"""

import unittest
import unittest.mock as mock
import json
import os
import numpy as np
import requests  # Importar para usar sus excepciones

# Importar la clase y constantes/funciones necesarias
from agent_ai.agent_ai import (AgentAI,
                               HARMONY_CONTROLLER_URL_ENV,
                               HARMONY_CONTROLLER_REGISTER_URL_ENV,
                               AGENT_AI_ECU_URL_ENV,
                               AGENT_AI_MALLA_URL_ENV, DEFAULT_HC_URL,
                               DEFAULT_ECU_URL, DEFAULT_MALLA_URL,
                               GLOBAL_REQUIREMENTS_PATH, REQUESTS_TIMEOUT,
                               MAX_RETRIES)


@mock.patch("agent_ai.agent_ai.requests", new_callable=mock.MagicMock)
@mock.patch("agent_ai.agent_ai.check_missing_dependencies")
@mock.patch("agent_ai.agent_ai.validate_module_registration")
@mock.patch("agent_ai.agent_ai.get_logger")
class TestAgentAIStrategicLogic(unittest.TestCase):
    """Suite de pruebas para la clase AgentAI."""

    def setUp(self):
        """
        Configura el entorno para cada prueba, creando una nueva instancia
        de AgentAI.
        """
        # Limpiar variables de entorno para aislar las pruebas
        os.environ.pop(HARMONY_CONTROLLER_URL_ENV, None)
        os.environ.pop(HARMONY_CONTROLLER_REGISTER_URL_ENV, None)
        os.environ.pop(AGENT_AI_ECU_URL_ENV, None)
        os.environ.pop(AGENT_AI_MALLA_URL_ENV, None)
        os.environ.pop("AA_INITIAL_SETPOINT_VECTOR", None)
        os.environ.pop("AA_INITIAL_STRATEGY", None)

        self.agent = AgentAI()
        # Detener el bucle estratégico que se inicia en __init__
        # para que no interfiera con las pruebas unitarias.
        if self.agent._strategic_thread.is_alive():
            self.agent.shutdown()

    def tearDown(self):
        """
        Limpia después de cada prueba, asegurando que el hilo del agente
        esté detenido.
        """
        if self.agent._strategic_thread.is_alive():
            self.agent.shutdown()

    def test_initialization_defaults(
        self, mock_logger, mock_validate, mock_check_deps, mock_requests
    ):
        """
        Verifica que el agente se inicializa con valores por defecto
        correctos cuando no hay variables de entorno.
        """
        self.assertEqual(len(self.agent.modules), 0)
        self.assertEqual(self.agent.current_strategy, "default")
        self.assertListEqual(self.agent.target_setpoint_vector, [1.0, 0.0])
        self.assertEqual(
            self.agent.central_urls["harmony_controller"], DEFAULT_HC_URL
        )
        self.assertEqual(self.agent.central_urls["ecu"], DEFAULT_ECU_URL)
        self.assertEqual(
            self.agent.central_urls["malla_watcher"], DEFAULT_MALLA_URL
        )

    # --- NUEVOS TESTS para verificar __init__ y lectura de ENV ---

    def test_init_central_urls_defaults(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que __init__ usa URLs default si ENV no están definidas."""
        # Asegurar un entorno limpio para este test, sin las variables
        # relevantes
        env_vars_to_clear = [
            HARMONY_CONTROLLER_URL_ENV,
            HARMONY_CONTROLLER_REGISTER_URL_ENV,
            AGENT_AI_ECU_URL_ENV,
            AGENT_AI_MALLA_URL_ENV,
        ]
        # Usar patch.dict para modificar os.environ temporalmente
        with mock.patch.dict(
            os.environ, {k: "" for k in env_vars_to_clear}, clear=True
        ):
            # Crear instancia DENTRO del contexto del patch
            agent_test = AgentAI()

            # Verificar que las URLs almacenadas son las defaults
            self.assertEqual(
                agent_test.central_urls.get("harmony_controller"),
                DEFAULT_HC_URL,
            )
            self.assertEqual(
                agent_test.central_urls.get("ecu"), DEFAULT_ECU_URL
            )
            self.assertEqual(
                agent_test.central_urls.get("malla_watcher"), DEFAULT_MALLA_URL
            )
            # Verificar que la URL de registro se construye con el default de
            # HC
            self.assertEqual(
                agent_test.hc_register_url,
                f"{DEFAULT_HC_URL}/api/harmony/register_tool",
            )

    def test_init_central_urls_from_env(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que __init__ usa URLs definidas en variables de entorno."""
        # Definir URLs de prueba para el entorno
        test_hc_url = "http://test-hc:111"
        test_ecu_url = "http://test-ecu:222"
        test_malla_url = "http://test-malla:333"
        test_hc_reg_url = "http://test-hc:111/custom_register"

        # Simular las variables de entorno
        simulated_env = {
            HARMONY_CONTROLLER_URL_ENV: test_hc_url,
            AGENT_AI_ECU_URL_ENV: test_ecu_url,
            AGENT_AI_MALLA_URL_ENV: test_malla_url,
            HARMONY_CONTROLLER_REGISTER_URL_ENV: test_hc_reg_url,
        }
        with mock.patch.dict(os.environ, simulated_env, clear=True):
            # Crear instancia DENTRO del contexto del patch
            agent_test = AgentAI()

            # Verificar que las URLs almacenadas son las del entorno simulado
            self.assertEqual(
                agent_test.central_urls.get("harmony_controller"), test_hc_url
            )
            self.assertEqual(agent_test.central_urls.get("ecu"), test_ecu_url)
            self.assertEqual(
                agent_test.central_urls.get("malla_watcher"), test_malla_url
            )
            # Verificar que la URL de registro es la del entorno simulado
            self.assertEqual(agent_test.hc_register_url, test_hc_reg_url)

    # --- FIN NUEVOS TESTS ---
    # --- Tests de Comunicación con HC (sin cambios funcionales) ---
    def test_get_harmony_state_success(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba obtener estado de Harmony con éxito."""
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_harmony_data = {
            "last_measurement": 0.5,
            "last_ecu_state": [[0.4], [0.3]],
        }  # Ejemplo con lista de listas
        mock_response.json.return_value = {
            "status": "success",
            "data": mock_harmony_data,
        }
        mock_requests.get.return_value = mock_response
        mock_requests.get.side_effect = None  # Limpiar side effects previos
        state = self.agent._get_harmony_state()
        hc_url = self.agent.central_urls.get(
            'harmony_controller', DEFAULT_HC_URL
        )
        expected_url = f"{hc_url}/api/harmony/state"
        mock_requests.get.assert_called_once_with(
            expected_url, timeout=REQUESTS_TIMEOUT  # Usar la URL esperada
        )
        self.assertEqual(state, mock_harmony_data)

    def test_get_harmony_state_network_error(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba obtener estado de Harmony con error de red y reintentos."""
        mock_requests.get.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )
        state = self.agent._get_harmony_state()
        self.assertIsNone(
            state, "Debe devolver None en caso de error de red persistente"
        )
        # Verificar que reintentó MAX_RETRIES veces
        self.assertEqual(mock_requests.get.call_count, MAX_RETRIES)
        # Verificar que se llamó a sleep (mockeado) entre intentos
        self.assertEqual(self.mock_sleep.call_count, MAX_RETRIES - 1)

    def test_get_harmony_state_bad_response(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba obtener estado de Harmony con respuesta inválida."""
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "error",
            "message": "Internal Server Error",
        }
        mock_requests.get.return_value = mock_response
        with mock.patch("agent_ai.agent_ai.MAX_RETRIES", 1):
            state = self.agent._get_harmony_state()
            self.assertIsNone(state)

    def test_send_setpoint_to_harmony_success(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba enviar setpoint a Harmony con éxito."""
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_requests.post.return_value = mock_response
        mock_requests.post.side_effect = None
        setpoint_vec = [1.5, -0.5]
        self.agent._send_setpoint_to_harmony(setpoint_vec)
        hc_url = self.agent.central_urls.get(
            'harmony_controller', DEFAULT_HC_URL
        )
        expected_url = f"{hc_url}/api/harmony/setpoint"
        mock_requests.post.assert_called_once_with(
            expected_url,  # Usar la URL esperada
            json={"setpoint_vector": setpoint_vec},
            timeout=REQUESTS_TIMEOUT,
        )

    def test_send_setpoint_to_harmony_error(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba enviar setpoint a Harmony con error de red y reintentos."""
        mock_requests.post.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )
        setpoint_vec = [1.5, -0.5]
        self.agent._send_setpoint_to_harmony(setpoint_vec)
        # Verificar que reintentó MAX_RETRIES veces
        self.assertEqual(mock_requests.post.call_count, MAX_RETRIES)
        self.assertEqual(self.mock_sleep.call_count, MAX_RETRIES - 1)

    def test_determine_harmony_setpoint_simple(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba la lógica refinada de determinación de setpoint."""

        # Caso default: sin cambios
        current_target = [1.0, 0.0]
        self.agent.target_setpoint_vector = list(current_target)
        self.agent.external_inputs["cogniboard_signal"] = None
        new_sp = self.agent._determine_harmony_setpoint(
            0.5, None, None, "default", {}
        )
        np.testing.assert_allclose(
            new_sp,
            current_target,
            err_msg="Default: El setpoint no debe cambiar.",
        )

        # Caso cogniboard señal alta: reducción
        current_target = [3.0, 4.0]
        scale = 0.9
        expected_target = [3.0 * scale, 4.0 * scale]
        self.agent.target_setpoint_vector = list(current_target)
        self.agent.external_inputs["cogniboard_signal"] = 0.9
        new_sp = self.agent._determine_harmony_setpoint(
            0.5, 0.9, None, "default", {}
        )
        np.testing.assert_allclose(
            new_sp,
            expected_target,
            err_msg="Cogniboard: El setpoint no se redujo correctamente.",
        )

        # Caso cogniboard señal inválida: sin cambios
        current_target = [1.0, 0.0]
        self.agent.target_setpoint_vector = list(current_target)
        self.agent.external_inputs["cogniboard_signal"] = "invalid"
        new_sp = self.agent._determine_harmony_setpoint(
            0.5, "invalid", None, "default", {}
        )
        np.testing.assert_allclose(
            new_sp,
            current_target,
            err_msg="Cogniboard inválido: El setpoint no debe cambiar.",
        )

        # Estrategia: estabilidad, reducción por error bajo o esfuerzo alto
        self.agent.target_setpoint_vector = [2.0, 0.0]
        modules = {}
        # Simular error bajo
        new_sp = self.agent._determine_harmony_setpoint(
            1.99, None, None, "estabilidad", modules
        )
        self.assertTrue(
            np.linalg.norm(new_sp) < np.linalg.norm([2.0, 0.0]),
            "Estabilidad: Debe reducir magnitud.",
        )

        # Estrategia: estabilidad, reducción extra por más reductores que
        # potenciadores
        modules = {
            "aux1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "reductor",
            },
            "aux2": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "reductor",
            },
            "aux3": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "potenciador",
            },
        }
        self.agent.target_setpoint_vector = [2.0, 0.0]
        new_sp = self.agent._determine_harmony_setpoint(
            1.99, None, None, "estabilidad", modules
        )
        # Debe reducir más que solo por el primer ajuste
        self.assertTrue(
            np.linalg.norm(new_sp) < 2.0 * 0.98,
            "Estabilidad: Reducción extra por reductores.",
        )

        # Estrategia: rendimiento, aumento por error bajo y esfuerzo bajo
        self.agent.target_setpoint_vector = [2.0, 0.0]
        self.agent.harmony_state["last_pid_output"] = 0.1
        new_sp = self.agent._determine_harmony_setpoint(
            1.99, None, None, "rendimiento", {}
        )
        self.assertTrue(
            np.linalg.norm(new_sp) > 2.0,
            "Rendimiento: Debe aumentar magnitud.",
        )

        # Estrategia: rendimiento, aumento extra por más potenciadores en ECU
        modules = {
            "aux1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "matriz_ecu",
                "naturaleza_auxiliar": "potenciador",
            },
            "aux2": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "matriz_ecu",
                "naturaleza_auxiliar": "potenciador",
            },
            "aux3": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "matriz_ecu",
                "naturaleza_auxiliar": "reductor",
            },
        }
        self.agent.target_setpoint_vector = [2.0, 0.0]
        self.agent.harmony_state["last_pid_output"] = 0.1
        new_sp = self.agent._determine_harmony_setpoint(
            1.99, None, None, "rendimiento", modules
        )
        # Debe aumentar más que solo por el primer ajuste
        self.assertTrue(
            np.linalg.norm(new_sp) > 2.0 * 1.02,
            "Rendimiento: Aumento extra por potenciadores ECU.",
        )

        # Estrategia: ahorro_energia, reducción si hay reductores activos
        modules = {
            "aux1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "reductor",
            },
            "aux2": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "matriz_ecu",
                "naturaleza_auxiliar": "reductor",
            },
        }
        self.agent.target_setpoint_vector = [2.0, 0.0]
        new_sp = self.agent._determine_harmony_setpoint(
            1.99, None, None, "ahorro_energia", modules
        )
        self.assertTrue(
            np.linalg.norm(new_sp) < 2.0,
            "Ahorro energía: Debe reducir magnitud si hay reductores.",
        )

        # --- NUEVOS TESTS DETALLADOS --- #

    def test_determine_estrategia_default_sin_cambio_base(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que la estrategia 'default' no cambia el setpoint sin otros factores."""
        initial_vector = [1.5, -0.5]
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": np.linalg.norm(initial_vector),
            "last_pid_output": 0.1,
        }
        modules = {}  # Sin auxiliares activos
        measurement = np.linalg.norm(
            initial_vector
        )  # Simular medición igual al setpoint

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "default", modules
        )

        np.testing.assert_allclose(
            new_sp,
            initial_vector,
            err_msg="Default: Setpoint no debería cambiar.",
        )

    def test_determine_estrategia_estabilidad_reduce_por_error_bajo(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que 'estabilidad' reduce magnitud si el error es bajo."""
        initial_vector = [2.0, 0.0]
        initial_norm = np.linalg.norm(initial_vector)
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.1,
        }  # PID bajo
        modules = {}
        measurement = (
            initial_norm * 0.95
        )  # Error bajo (measurement cercano a setpoint)

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "estabilidad", modules
        )
        final_norm = np.linalg.norm(new_sp)

        self.assertTrue(
            final_norm < initial_norm,
            "Estabilidad/Error Bajo: Norma debería reducirse.",
        )
        # Verificar que la reducción es la esperada (0.98)
        np.testing.assert_allclose(final_norm, initial_norm * 0.98, rtol=1e-6)

    def test_determine_estrategia_estabilidad_reduce_por_pid_alto(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que 'estabilidad' reduce magnitud si el esfuerzo PID es alto."""
        initial_vector = [2.0, 0.0]
        initial_norm = np.linalg.norm(initial_vector)
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.8,
        }  # PID alto
        modules = {}
        measurement = (
            initial_norm * 0.5
        )  # Error grande (no debería importar para esta condición)

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "estabilidad", modules
        )
        final_norm = np.linalg.norm(new_sp)

        self.assertTrue(
            final_norm < initial_norm,
            "Estabilidad/PID Alto: Norma debería reducirse.",
        )
        np.testing.assert_allclose(final_norm, initial_norm * 0.98, rtol=1e-6)

    def test_determine_estrategia_estabilidad_reduce_extra_por_reductores(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica reducción extra en 'estabilidad' si hay más reductores en malla."""
        initial_vector = [2.0, 0.0]
        initial_norm = np.linalg.norm(initial_vector)
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.1,
        }  # PID bajo
        modules = {
            "r1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "reductor",
            },
            "r2": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "reductor",
            },
            "p1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "potenciador",
            },
        }
        measurement = initial_norm * 0.95  # Error bajo

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "estabilidad", modules
        )
        final_norm = np.linalg.norm(new_sp)

        # Se aplican ambas reducciones: 0.98 * 0.97
        expected_norm = initial_norm * 0.98 * 0.97
        self.assertTrue(
            final_norm < initial_norm * 0.98,
            "Estabilidad/Reductores: Reducción extra esperada.",
        )
        np.testing.assert_allclose(final_norm, expected_norm, rtol=1e-6)

    def test_determine_estrategia_rendimiento_aumenta(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que 'rendimiento' aumenta magnitud si está estable."""
        initial_vector = [2.0, 0.0]
        initial_norm = np.linalg.norm(initial_vector)
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.1,
        }  # PID bajo
        modules = {}
        measurement = initial_norm * 0.99  # Error bajo

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "rendimiento", modules
        )
        final_norm = np.linalg.norm(new_sp)

        self.assertTrue(
            final_norm > initial_norm,
            "Rendimiento/Estable: Norma debería aumentar.",
        )
        np.testing.assert_allclose(final_norm, initial_norm * 1.02, rtol=1e-6)

    def test_determine_estrategia_rendimiento_aumenta_extra_por_potenciadores(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica aumento extra en 'rendimiento' si hay más potenciadores en ECU."""
        initial_vector = [2.0, 0.0]
        initial_norm = np.linalg.norm(initial_vector)
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.1,
        }  # PID bajo
        modules = {
            "p1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "matriz_ecu",
                "naturaleza_auxiliar": "potenciador",
            },
            "p2": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "matriz_ecu",
                "naturaleza_auxiliar": "potenciador",
            },
            "r1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "matriz_ecu",
                "naturaleza_auxiliar": "reductor",
            },
        }
        measurement = initial_norm * 0.99  # Error bajo

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "rendimiento", modules
        )
        final_norm = np.linalg.norm(new_sp)

        # Se aplican ambos aumentos: 1.02 * 1.01
        expected_norm = initial_norm * 1.02 * 1.01
        self.assertTrue(
            final_norm > initial_norm * 1.02,
            "Rendimiento/Potenciadores: Aumento extra esperado.",
        )
        np.testing.assert_allclose(final_norm, expected_norm, rtol=1e-6)

    def test_determine_estrategia_rendimiento_establece_minimo_si_cero(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que 'rendimiento' establece un mínimo si el setpoint es cero."""
        initial_vector = [0.0, 0.0]  # Setpoint inicial cero
        initial_norm = 0.0
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.1,
        }  # PID bajo
        modules = {}
        measurement = 0.0  # Error cero

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "rendimiento", modules
        )
        final_norm = np.linalg.norm(new_sp)

        self.assertTrue(
            final_norm > 0, "Rendimiento/Cero: Norma debería ser > 0."
        )
        # Verificar que establece el mínimo [0.1, 0.1] (asumiendo vector 2D)
        expected_vector = [0.1] * len(initial_vector)
        np.testing.assert_allclose(new_sp, expected_vector)

    def test_determine_estrategia_ahorro_reduce_con_reductores(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que 'ahorro_energia' reduce magnitud si hay reductores."""
        initial_vector = [2.0, 0.0]
        initial_norm = np.linalg.norm(initial_vector)
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.1,
        }
        modules = {
            "r1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "reductor",
            }
        }
        measurement = initial_norm  # Sin error

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "ahorro_energia", modules
        )
        final_norm = np.linalg.norm(new_sp)

        self.assertTrue(
            final_norm < initial_norm,
            "Ahorro/Reductores: Norma debería reducirse.",
        )
        np.testing.assert_allclose(final_norm, initial_norm * 0.95, rtol=1e-6)

    def test_determine_estrategia_ahorro_sin_cambio_sin_reductores(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que 'ahorro_energia' no cambia si no hay reductores."""
        initial_vector = [2.0, 0.0]
        initial_norm = np.linalg.norm(initial_vector)
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.1,
        }
        modules = {  # Sin reductores activos
            "p1": {
                "tipo": "auxiliar",
                "estado_salud": "ok",
                "aporta_a": "malla_watcher",
                "naturaleza_auxiliar": "potenciador",
            }
        }
        measurement = initial_norm  # Sin error

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, None, None, "ahorro_energia", modules
        )

        np.testing.assert_allclose(
            new_sp,
            initial_vector,
            err_msg="Ahorro/Sin Reductores: Setpoint no debería cambiar.",
        )

    def test_determine_cogniboard_alto_reduce(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Verifica que la señal alta de cogniboard reduce la magnitud final."""
        initial_vector = [5.0, 0.0]
        initial_norm = np.linalg.norm(initial_vector)
        self.agent.target_setpoint_vector = list(initial_vector)
        self.agent.harmony_state = {
            "setpoint_value": initial_norm,
            "last_pid_output": 0.1,
        }
        modules = {}
        measurement = initial_norm  # Sin error
        cogniboard_signal = 0.9  # Señal alta

        new_sp = self.agent._determine_harmony_setpoint(
            measurement, cogniboard_signal, None, "default", modules
        )
        final_norm = np.linalg.norm(new_sp)

        self.assertTrue(
            final_norm < initial_norm,
            "Cogniboard Alto: Norma debería reducirse.",
        )
        np.testing.assert_allclose(final_norm, initial_norm * 0.9, rtol=1e-6)
    # --- FIN NUEVOS TESTS DETALLADOS ---

    # --- Tests de Registro de Módulos (AJUSTADOS) ---

    # --- MODIFICADO: Incluir tipo, aporta_a, naturaleza_auxiliar ---
    def test_registrar_modulo_auxiliar_success(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba registro exitoso de módulo auxiliar con afinidad y naturaleza."""
        mock_check_deps.return_value = (True, "Dependencias OK")
        mock_os_exists.return_value = True  # Asumir que archivos existen

        module_data = {
            "nombre": "AuxTest",
            "url": "http://auxtest:1234/api/state",  # URL de control
            "url_salud": "http://auxtest:1234/api/health",
            "tipo": "auxiliar",
            "aporta_a": "malla_watcher",
            "naturaleza_auxiliar": "potenciador",
            "requirements_path": "dummy_req.txt",
        }
        result = self.agent.registrar_modulo(module_data)

        self.assertEqual(
            result["status"], "success", f"Resultado inesperado: {result}"
        )
        self.assertIn("AuxTest", self.agent.modules)
        module_entry = self.agent.modules["AuxTest"]
        self.assertEqual(module_entry["estado_salud"], "pendiente")
        self.assertEqual(module_entry["tipo"], "auxiliar")
        self.assertEqual(module_entry["aporta_a"], "malla_watcher")
        self.assertEqual(module_entry["naturaleza_auxiliar"], "potenciador")
        self.assertEqual(
            module_entry["url"], "http://auxtest:1234/api/state"
        )
        self.assertEqual(
            module_entry["url_salud"],
            "http://auxtest:1234/api/health"
        )

        # Verificar que se intentó iniciar el hilo de validación
        mock_thread.assert_called_once()
        # Verificar que check_missing_dependencies fue llamado
        mock_check_deps.assert_called_once_with(
            "dummy_req.txt", GLOBAL_REQUIREMENTS_PATH
        )

    def test_registrar_modulo_central_success(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba registro exitoso de módulo central (sin afinidad/naturaleza)."""
        mock_check_deps.return_value = (True, "Dependencias OK")
        mock_os_exists.return_value = True

        module_data = {
            "nombre": "CentralTest",
            "url": "http://centraltest:5678/api/state",
            "url_salud": "http://centraltest:5678/api/health",
            "tipo": "central",  # O integrador, convergente si se usa esa clasificación aquí
            # No se proporcionan aporta_a ni naturaleza_auxiliar
        }
        result = self.agent.registrar_modulo(module_data)

        self.assertEqual(result["status"], "success")
        self.assertIn("CentralTest", self.agent.modules)
        module_entry = self.agent.modules["CentralTest"]
        self.assertEqual(module_entry["tipo"], "central")
        self.assertNotIn("aporta_a", module_entry)
        self.assertNotIn("naturaleza_auxiliar", module_entry)
        mock_thread.assert_called_once()  # Hilo de validación se inicia igual

    def test_registrar_modulo_invalid_data(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba registro con datos faltantes (validator se encarga)."""
        # Asumiendo que validator.validate_module_registration verifica campos
        # requeridos
        module_data = {"nombre": "TestReg"}  # Falta url, tipo, etc.
        # Mockear el validador para simular fallo
        with mock.patch(
            "agent_ai.agent_ai.validate_module_registration",
            return_value=(False, "Faltan campos requeridos: url, tipo"),
        ):
            result = self.agent.registrar_modulo(module_data)
        self.assertEqual(result["status"], "error")
        self.assertIn("Faltan campos requeridos", result["mensaje"])
        self.assertNotIn("TestReg", self.agent.modules)

    def test_registrar_modulo_dep_fail(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba registro con fallo de dependencias."""
        mock_check_deps.return_value = (False, "Falta 'superlib'")
        mock_os_exists.return_value = True
        module_data = {
            "nombre": "TestDepFail",
            "url": "http://testdep/health",
            "tipo": "auxiliar",
            "requirements_path": "req.txt",
        }
        result = self.agent.registrar_modulo(module_data)
        self.assertEqual(result["status"], "error")
        self.assertIn("Falta 'superlib'", result["mensaje"])
        self.assertNotIn("TestDepFail", self.agent.modules)

    def test_validar_salud_modulo_ok_auxiliar_sin_naturaleza(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba validación OK de auxiliar pero sin naturaleza (no notifica)."""
        module_name = "NoNatureAux"
        module_url = "http://nonature/api"
        # Configurar módulo sin naturaleza_auxiliar
        self.agent.modules[module_name] = {
            "nombre": module_name,
            "url": module_url,
            "url_salud": module_url,
            "tipo": "auxiliar",
            "aporta_a": "malla_watcher",  # Falta naturaleza
            "estado_salud": "pendiente",
        }
        mock_health_resp = mock.MagicMock(status_code=200)
        mock_requests.get.return_value = mock_health_resp
        mock_requests.get.side_effect = None

        self.agent._validar_salud_modulo(module_name)

        self.assertEqual(self.agent.modules[module_name]["estado_salud"], "ok")
        mock_requests.get.assert_called_once_with(
            module_url, timeout=REQUESTS_TIMEOUT
        )
        # Verificar que NO se llamó a post para notificar a HC
        mock_requests.post.assert_not_called()

    def test_validar_salud_modulo_ok_central(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba validación OK de central (no notifica a HC)."""
        module_name = "HealthyCentral"
        module_url = "http://healthycentral/api"
        self.agent.modules[module_name] = {
            "nombre": module_name,
            "url": module_url,
            "url_salud": module_url,
            "tipo": "central",  # O integrador, etc.
            "estado_salud": "pendiente",
        }
        mock_health_resp = mock.MagicMock(status_code=200)
        mock_requests.get.return_value = mock_health_resp
        mock_requests.get.side_effect = None

        self.agent._validar_salud_modulo(module_name)

        self.assertEqual(self.agent.modules[module_name]["estado_salud"], "ok")
        mock_requests.get.assert_called_once_with(
            module_url, timeout=REQUESTS_TIMEOUT
        )
        mock_requests.post.assert_not_called()  # No debe notificar centrales

    def test_validar_salud_modulo_fail_y_no_notifica(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba validación fallida (no debe notificar)."""
        module_name = "FailAux"
        module_url = "http://failaux/api"
        self.agent.modules[module_name] = {
            "nombre": module_name,
            "url": module_url,
            "url_salud": module_url,
            "tipo": "auxiliar",
            "aporta_a": "matriz_ecu",
            "naturaleza_auxiliar": "potenciador",
            "estado_salud": "pendiente",
        }
        mock_requests.get.side_effect = requests.exceptions.ConnectionError(
            "Fail"
        )

        self.agent._validar_salud_modulo(module_name)

        self.assertEqual(
            self.agent.modules[module_name]["estado_salud"], "error_inesperado"
        )
        # Verificar que reintentó
        self.assertEqual(mock_requests.get.call_count, MAX_RETRIES)
        # Verificar que NO notificó
        mock_requests.post.assert_not_called()

    def test_notify_hc_retry_and_fail(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba que la notificación a HC reintenta y falla."""
        # Mockear la notificación para que siempre falle
        mock_requests.post.side_effect = requests.exceptions.RequestException(
            "HC down"
        )

        # Llamar directamente a la función de notificación
        self.agent._notify_harmony_controller_of_tool(
            nombre="NotifyFail",
            url="http://someurl",
            aporta_a="malla",
            naturaleza="modulador",
        )

        # Verificar reintentos
        self.assertEqual(mock_requests.post.call_count, MAX_RETRIES)
        self.assertEqual(self.mock_sleep.call_count, MAX_RETRIES - 1)

    # --- Tests de Comandos y Estado Completo (AJUSTADOS) ---

    def test_actualizar_comando_estrategico(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba comandos estratégicos."""
        # Set strategy
        result = self.agent.actualizar_comando_estrategico(
            "set_strategy", "performance"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.agent.current_strategy, "performance")

        # Set setpoint vector
        mock_requests.post.reset_mock()
        mock_response_post = mock.MagicMock(status_code=200)
        mock_requests.post.return_value = mock_response_post
        mock_requests.post.side_effect = None

        # --- ASEGÚRATE DE QUE ESTA LÍNEA ESTÉ PRESENTE ---
        new_vec = [2.0, 1.0]
        # -------------------------------------------------
        result = self.agent.actualizar_comando_estrategico(
            "set_target_setpoint_vector", new_vec
        )  # Ahora new_vec existe
        self.assertEqual(result["status"], "success")
        self.assertEqual(self.agent.target_setpoint_vector, new_vec)
        # ... (verificación de llamada a post) ...
        hc_url = self.agent.central_urls.get(
            'harmony_controller', DEFAULT_HC_URL
        )
        expected_url = f"{hc_url}/api/harmony/setpoint"
        mock_requests.post.assert_called_once_with(
            expected_url,
            json={"setpoint_vector": new_vec},
            timeout=REQUESTS_TIMEOUT,
        )

        # Comando inválido
        result = self.agent.actualizar_comando_estrategico("invalid_cmd", None)
        self.assertEqual(result["status"], "error")
        self.assertIn("no reconocido", result["mensaje"])

    def test_recibir_inputs_externos(
        self, mock_thread, mock_os_exists, mock_check_deps, mock_requests
    ):
        """Prueba la recepción de señales externas."""
        self.agent.recibir_control_cogniboard(0.77)
        self.assertEqual(self.agent.external_inputs["cogniboard_signal"], 0.77)
        config_data = {"status": "healthy"}
        self.agent.recibir_config_status(config_data)
        self.assertEqual(
            self.agent.external_inputs["config_status"], config_data
        )

    # --- MODIFICADO: test_obtener_estado_completo incluye naturaleza ---
    def test_get_full_state_snapshot(
        self, mock_logger, mock_validate, mock_check_deps, mock_requests
    ):
        """
        Verifica que obtener_estado_completo retorna una instantánea
        correcta y completa del estado del agente.
        """
        # Configurar estado inicial
        with self.agent.lock:
            self.agent.target_setpoint_vector = [0.5, -0.5]
            self.agent.current_strategy = "test_strat"
            self.agent.external_inputs["cogniboard_signal"] = 0.1
            self.agent.harmony_state = {"last_measurement": 0.9}
            self.agent.modules = {
                "TestModCentral": {
                    "tipo": "integrador",
                    "estado_salud": "error_timeout",
                },
                "TestModAux": {
                    "tipo": "auxiliar",
                    "aporta_a": "malla_watcher",
                    "naturaleza_auxiliar": "modulador",
                    "estado_salud": "ok",
                },
            }

        estado = self.agent.obtener_estado_completo()

        # SOLUCIÓN E501: Se divide la aserción en múltiples líneas para claridad.
        self.assertListEqual(
            estado["target_setpoint_vector"], [0.5, -0.5]
        )
        self.assertEqual(estado["current_strategy"], "test_strat")
        self.assertEqual(
            estado["external_inputs"]["cogniboard_signal"], 0.1
        )
        # SOLUCIÓN E501
        self.assertEqual(
            estado["harmony_controller_last_state"]["last_measurement"],
            0.9,
        )
        self.assertEqual(len(estado["registered_modules"]), 2)

        # Verificar detalles del módulo auxiliar
        mod_aux = next(
            m
            for m in estado["registered_modules"]
            if m["nombre"] == "TestModAux"
        )
        self.assertEqual(mod_aux["tipo"], "auxiliar")
        self.assertEqual(mod_aux["aporta_a"], "malla_watcher")
        # SOLUCIÓN E501
        self.assertEqual(
            mod_aux["naturaleza_auxiliar"], "modulador"
        )
        self.assertEqual(mod_aux["estado_salud"], "ok")

        # Verificar detalles del módulo central
        mod_central = next(
            m
            for m in estado["registered_modules"]
            if m["nombre"] == "TestModCentral"
        )
        self.assertEqual(mod_central["tipo"], "integrador")
        # SOLUCIÓN E501
        self.assertNotIn("aporta_a", mod_central)
        self.assertNotIn("naturaleza_auxiliar", mod_central)
        self.assertEqual(mod_central["estado_salud"], "error_timeout")


if __name__ == "__main__":
    unittest.main()

# --- END OF FILE tests/unit/test_agent_ai.py (AJUSTADO Fase 1) ---