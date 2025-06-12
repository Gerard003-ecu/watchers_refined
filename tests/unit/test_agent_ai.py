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

# Importar la clase y constantes/funciones necesarias desde la
# ubicación correcta
# Asumiendo que la estructura es mi-proyecto/agent_ai/agent_ai.py
from agent_ai.agent_ai import (AgentAI,
                               HARMONY_CONTROLLER_URL_ENV,
                               HARMONY_CONTROLLER_REGISTER_URL_ENV,
                               AGENT_AI_ECU_URL_ENV,
                               AGENT_AI_MALLA_URL_ENV, DEFAULT_HC_URL,
                               DEFAULT_ECU_URL, DEFAULT_MALLA_URL,
                               GLOBAL_REQUIREMENTS_PATH, REQUESTS_TIMEOUT,
                               MAX_RETRIES)
# STRATEGIC_LOOP_INTERVAL,  # F401 - Unused
# BASE_RETRY_DELAY,  # F401 - Unused


# Importar funciones/clases a mockear desde su ubicación real
# from agent_ai.validation.validator import ( # F401 - Unused
#     validate_module_registration,
#     check_missing_dependencies,
# )


# Mockear requests globalmente para la clase
@mock.patch("agent_ai.agent_ai.requests", new_callable=mock.MagicMock)
@mock.patch("agent_ai.agent_ai.check_missing_dependencies")
@mock.patch("agent_ai.agent_ai.os.path.exists")
@mock.patch("agent_ai.agent_ai.threading.Thread")
class TestAgentAIStrategicLogic(unittest.TestCase):

    def setUp(self):
        """Crea una nueva instancia de AgentAI para cada test."""
        # Limpiar mocks residuales de tests anteriores si los hubiera
        mock.patch.stopall()
        # Crear instancia
        self.agent = AgentAI()
        # Detener bucle si se inició accidentalmente
        self.agent.stop_strategic_loop()

    def tearDown(self):
        """Limpia al terminar cada test."""
        # Detener bucle si se inició
        self.agent.stop_strategic_loop()

    def _configure_status_registry_ok(self,
                                      mock_thread, mock_exists, 
                                      mock_check_deps, mock_requests):
        """Helper común: configura mocks para que el registro pase OK."""
        # Archivo existe
        mock_exists.return_value = True
        # Sin dependencias faltantes
        mock_check_deps.return_value = []
        # Requests exitosos (POST register, GET status)
        response_register = mock.MagicMock()
        response_register.status_code = 200
        response_register.json.return_value = {"status": "registered"}
        
        response_status = mock.MagicMock()
        response_status.status_code = 200
        response_status.json.return_value = {
            "modules": [],
            "cogniboard_signal": 0.0
        }
        
        mock_requests.post.return_value = response_register
        mock_requests.get.return_value = response_status

    def _configure_validation_ok(self,
                                 mock_thread, mock_exists, 
                                 mock_check_deps, mock_requests):
        """Helper común: configura mocks para que la validación pase OK."""
        self._configure_status_registry_ok(mock_thread, mock_exists,
                                           mock_check_deps, mock_requests)

    def _configure_validation_error(self,
                                    mock_thread, mock_exists, 
                                    mock_check_deps, mock_requests):
        """Helper común: configura mocks para que la validación falle."""
        # Archivo NO existe
        mock_exists.return_value = False
        # O dependencias faltantes
        mock_check_deps.return_value = ["missing_dep"]

    def test_init_default_urls(self,
                               mock_thread, mock_exists, 
                               mock_check_deps, mock_requests):
        """Verifica URLs por defecto si no están en environment."""
        # Asegurar que environment no tiene estas variables
        os.environ.pop(HARMONY_CONTROLLER_URL_ENV, None)
        os.environ.pop(AGENT_AI_ECU_URL_ENV, None)
        os.environ.pop(AGENT_AI_MALLA_URL_ENV, None)
        
        agent = AgentAI()
        
        self.assertEqual(agent.harmony_controller_url, DEFAULT_HC_URL)
        self.assertEqual(agent.ecu_url, DEFAULT_ECU_URL)
        self.assertEqual(agent.malla_url, DEFAULT_MALLA_URL)

    def test_init_environment_urls(self,
                                   mock_thread, mock_exists, 
                                   mock_check_deps, mock_requests):
        """Verifica URLs desde environment si están configuradas."""
        test_hc_url = "http://test.harmony:9999"
        test_ecu_url = "http://test.ecu:7777"
        test_malla_url = "http://test.malla:6666"
        
        os.environ[HARMONY_CONTROLLER_URL_ENV] = test_hc_url
        os.environ[AGENT_AI_ECU_URL_ENV] = test_ecu_url
        os.environ[AGENT_AI_MALLA_URL_ENV] = test_malla_url
        
        agent = AgentAI()
        
        self.assertEqual(agent.harmony_controller_url, test_hc_url)
        self.assertEqual(agent.ecu_url, test_ecu_url)
        self.assertEqual(agent.malla_url, test_malla_url)
        
        # Limpiar
        del os.environ[HARMONY_CONTROLLER_URL_ENV]
        del os.environ[AGENT_AI_ECU_URL_ENV]
        del os.environ[AGENT_AI_MALLA_URL_ENV]

    def test_initial_state(self,
                           mock_thread, mock_exists, 
                           mock_check_deps, mock_requests):
        """Verifica estado inicial del AgentAI."""
        self.assertIsInstance(self.agent.setpoint, float)
        self.assertEqual(self.agent.setpoint, 0.0)
        self.assertEqual(self.agent.strategy, "default")
        self.assertFalse(self.agent.strategic_loop_running)

    def test_register_success(self,
                              mock_thread, mock_exists, 
                              mock_check_deps, mock_requests):
        """Verifica registro exitoso con Harmony Controller."""
        self._configure_status_registry_ok(mock_thread, mock_exists,
                                           mock_check_deps, mock_requests)
        
        result = self.agent.register()
        
        self.assertTrue(result)
        # Verifica que se hizo el POST de registro
        mock_requests.post.assert_called()

    def test_register_request_failure(self,
                                      mock_thread, mock_exists, 
                                      mock_check_deps, mock_requests):
        """Verifica manejo de falla en request de registro."""
        mock_exists.return_value = True
        mock_check_deps.return_value = []
        # Falla en el POST
        mock_requests.post.side_effect = requests.exceptions.RequestException()
        
        result = self.agent.register()
        
        self.assertFalse(result)

    def test_register_validation_error(self,
                                       mock_thread, mock_exists, 
                                       mock_check_deps, mock_requests):
        """Verifica falla de registro por validación."""
        self._configure_validation_error(mock_thread, mock_exists,
                                         mock_check_deps, mock_requests)
        
        result = self.agent.register()
        
        self.assertFalse(result)

    def test_get_status_success(self,
                                mock_thread, mock_exists, 
                                mock_check_deps, mock_requests):
        """Verifica obtención exitosa de status."""
        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "modules": [{"id": "mod1"}],
            "cogniboard_signal": 0.5
        }
        mock_requests.get.return_value = response
        
        status = self.agent.get_status()
        
        self.assertIsNotNone(status)
        self.assertEqual(status["cogniboard_signal"], 0.5)

    def test_get_status_failure(self,
                                mock_thread, mock_exists, 
                                mock_check_deps, mock_requests):
        """Verifica manejo de falla en get_status."""
        mock_requests.get.side_effect = requests.exceptions.RequestException()
        
        status = self.agent.get_status()
        
        self.assertIsNone(status)

    def test_start_strategic_loop_success(self,
                                          mock_thread, mock_exists, 
                                          mock_check_deps, mock_requests):
        """Verifica inicio exitoso del loop estratégico."""
        self._configure_status_registry_ok(mock_thread, mock_exists,
                                           mock_check_deps, mock_requests)
        
        result = self.agent.start_strategic_loop()
        
        self.assertTrue(result)
        self.assertTrue(self.agent.strategic_loop_running)
        # Verifica que se creó el thread
        mock_thread.assert_called()

    def test_start_strategic_loop_already_running(self,
                                                  mock_thread, mock_exists, 
                                                  mock_check_deps, 
                                                  mock_requests):
        """Verifica que no inicia loop si ya está corriendo."""
        self.agent.strategic_loop_running = True
        
        result = self.agent.start_strategic_loop()
        
        self.assertFalse(result)
        # No debería crear thread nuevo
        mock_thread.assert_not_called()

    def test_start_strategic_loop_registration_fails(self,
                                                     mock_thread, mock_exists, 
                                                     mock_check_deps, 
                                                     mock_requests):
        """Verifica que no inicia loop si falla el registro."""
        self._configure_validation_error(mock_thread, mock_exists,
                                         mock_check_deps, mock_requests)
        
        result = self.agent.start_strategic_loop()
        
        self.assertFalse(result)
        self.assertFalse(self.agent.strategic_loop_running)

    def test_stop_strategic_loop(self,
                                 mock_thread, mock_exists, 
                                 mock_check_deps, mock_requests):
        """Verifica parada del loop estratégico."""
        self.agent.strategic_loop_running = True
        
        self.agent.stop_strategic_loop()
        
        self.assertFalse(self.agent.strategic_loop_running)

    def test_strategic_logic_cycle_success(self,
                                           mock_thread, mock_exists, 
                                           mock_check_deps, mock_requests):
        """Verifica ejecución exitosa de un ciclo de lógica estratégica."""
        # Configurar response del status
        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "modules": [
                {"id": "mod1", "tipo": "auxiliar", 
                 "rendimiento": 0.8, "naturaleza": "reductor"}
            ],
            "cogniboard_signal": 0.3
        }
        mock_requests.get.return_value = response
        
        # Ejecutar un ciclo
        self.agent._strategic_logic_cycle()
        
        # Verifica que se llamó get_status
        mock_requests.get.assert_called()

    def test_strategic_logic_cycle_get_status_fails(self,
                                                    mock_thread, mock_exists, 
                                                    mock_check_deps, 
                                                    mock_requests):
        """Verifica manejo de falla en get_status durante ciclo."""
        mock_requests.get.side_effect = requests.exceptions.RequestException()
        
        # No debería lanzar excepción
        try:
            self.agent._strategic_logic_cycle()
        except Exception as e:
            self.fail(f"_strategic_logic_cycle() lanzó excepción: {e}")

    def test_update_setpoint_strategy_default(self,
                                              mock_thread, mock_exists, 
                                              mock_check_deps, mock_requests):
        """Verifica que la estrategia 'default' no cambia el setpoint sin 
        otros factores."""
        setpoint_inicial = self.agent.setpoint
        modules = []
        cogniboard_signal = 0.0
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        self.assertEqual(self.agent.setpoint, setpoint_inicial)

    def test_update_setpoint_strategy_default_with_cogniboard(self,
                                                              mock_thread, 
                                                              mock_exists, 
                                                              mock_check_deps, 
                                                              mock_requests):
        """Verifica ajuste por cogniboard en estrategia default."""
        setpoint_inicial = 1.0
        self.agent.setpoint = setpoint_inicial
        modules = []
        cogniboard_signal = 0.8  # Alta
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        # Debería reducir el setpoint por señal alta de cogniboard
        self.assertLess(self.agent.setpoint, setpoint_inicial)

    def test_update_setpoint_strategy_estabilidad_high_pid(self,
                                                           mock_thread, 
                                                           mock_exists, 
                                                           mock_check_deps, 
                                                           mock_requests):
        """Verifica que 'estabilidad' reduce magnitud si el esfuerzo 
        PID es alto."""
        self.agent.strategy = "estabilidad"
        setpoint_inicial = 2.0
        self.agent.setpoint = setpoint_inicial
        
        # Simular alto esfuerzo PID con módulos de rendimiento variado
        modules = [
            {"id": "m1", "tipo": "auxiliar", "rendimiento": 0.2},  # Bajo
            {"id": "m2", "tipo": "auxiliar", "rendimiento": 0.9}   # Alto
        ]
        cogniboard_signal = 0.0
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        # Debería reducir magnitud
        self.assertLess(abs(self.agent.setpoint), abs(setpoint_inicial))

    def test_update_setpoint_strategy_estabilidad_with_reducers(self,
                                                                mock_thread, 
                                                                mock_exists, 
                                                                mock_check_deps, 
                                                                mock_requests):
        """Verifica reducción extra en 'estabilidad' si hay más 
        reductores en malla."""
        self.agent.strategy = "estabilidad"
        setpoint_inicial = 2.0
        self.agent.setpoint = setpoint_inicial
        
        modules = [
            {"id": "m1", "tipo": "auxiliar", "rendimiento": 0.5, 
             "naturaleza": "reductor"},
            {"id": "m2", "tipo": "auxiliar", "rendimiento": 0.5, 
             "naturaleza": "reductor"}
        ]
        cogniboard_signal = 0.0
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        # Con múltiples reductores, debería aplicar reducción extra
        self.assertLess(abs(self.agent.setpoint), abs(setpoint_inicial))

    def test_update_setpoint_strategy_rendimiento_low_pid(self,
                                                          mock_thread, 
                                                          mock_exists, 
                                                          mock_check_deps, 
                                                          mock_requests):
        """Verifica aumento en 'rendimiento' si el esfuerzo PID es bajo."""
        self.agent.strategy = "rendimiento"
        setpoint_inicial = 1.0
        self.agent.setpoint = setpoint_inicial
        
        # Simular bajo esfuerzo PID
        modules = [
            {"id": "m1", "tipo": "auxiliar", "rendimiento": 0.9},  # Alto
            {"id": "m2", "tipo": "auxiliar", "rendimiento": 0.8}   # Alto
        ]
        cogniboard_signal = 0.0
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        # Debería aumentar magnitud
        self.assertGreater(abs(self.agent.setpoint), abs(setpoint_inicial))

    def test_update_setpoint_strategy_rendimiento_with_enhancers(self,
                                                                 mock_thread, 
                                                                 mock_exists, 
                                                                 mock_check_deps, 
                                                                 mock_requests):
        """Verifica aumento extra en 'rendimiento' si hay más 
        potenciadores en ECU."""
        self.agent.strategy = "rendimiento"
        setpoint_inicial = 1.0
        self.agent.setpoint = setpoint_inicial
        
        modules = [
            {"id": "m1", "tipo": "central", "rendimiento": 0.8, 
             "naturaleza": "potenciador"},
            {"id": "m2", "tipo": "central", "rendimiento": 0.7, 
             "naturaleza": "potenciador"}
        ]
        cogniboard_signal = 0.0
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        # Con múltiples potenciadores, debería aplicar aumento extra
        self.assertGreater(abs(self.agent.setpoint), abs(setpoint_inicial))

    def test_update_setpoint_strategy_rendimiento_zero_setpoint(self,
                                                                mock_thread, 
                                                                mock_exists, 
                                                                mock_check_deps, 
                                                                mock_requests):
        """Verifica que 'rendimiento' establece un mínimo si el 
        setpoint es cero."""
        self.agent.strategy = "rendimiento"
        self.agent.setpoint = 0.0
        
        modules = [{"id": "m1", "tipo": "auxiliar", "rendimiento": 0.9}]
        cogniboard_signal = 0.0
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        # Debería establecer un mínimo positivo
        self.assertGreater(self.agent.setpoint, 0.0)

    def test_update_setpoint_strategy_unknown(self,
                                              mock_thread, mock_exists, 
                                              mock_check_deps, mock_requests):
        """Verifica comportamiento con estrategia desconocida."""
        self.agent.strategy = "estrategia_inexistente"
        setpoint_inicial = self.agent.setpoint
        
        modules = []
        cogniboard_signal = 0.0
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        # Debería mantener el setpoint sin cambios
        self.assertEqual(self.agent.setpoint, setpoint_inicial)

    def test_update_setpoint_cogniboard_high_signal_reduces(self,
                                                            mock_thread, 
                                                            mock_exists, 
                                                            mock_check_deps, 
                                                            mock_requests):
        """Verifica que la señal alta de cogniboard reduce la 
        magnitud final."""
        setpoint_inicial = 3.0
        self.agent.setpoint = setpoint_inicial
        
        modules = []
        cogniboard_signal = 0.9  # Muy alta
        
        self.agent._update_setpoint(modules, cogniboard_signal)
        
        # Debería reducir significativamente la magnitud
        self.assertLess(abs(self.agent.setpoint), abs(setpoint_inicial))

    def test_register_module_auxiliary_success(self,
                                               mock_thread, mock_exists, 
                                               mock_check_deps, mock_requests):
        """Prueba registro exitoso de módulo auxiliar con afinidad 
        y naturaleza."""
        module_info = {
            "id": "test_aux_module",
            "tipo": "auxiliar",
            "afinidad": "estabilidad",
            "naturaleza": "reductor"
        }
        
        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {"status": "module registered"}
        mock_requests.post.return_value = response
        
        result = self.agent.register_module(module_info)
        
        self.assertTrue(result)
        # Verifica que se hizo POST con info correcta
        mock_requests.post.assert_called_once()
        args, kwargs = mock_requests.post.call_args
        self.assertIn("json", kwargs)
        self.assertEqual(kwargs["json"]["id"], "test_aux_module")

    def test_register_module_central_success(self,
                                             mock_thread, mock_exists, 
                                             mock_check_deps, mock_requests):
        """Prueba registro exitoso de módulo central 
        (sin afinidad/naturaleza)."""
        module_info = {
            "id": "test_central_module",
            "tipo": "central",  # O integrador, convergente si se usa 
                                # esa clasificación aquí
        }
        
        response = mock.MagicMock()
        response.status_code = 200
        response.json.return_value = {"status": "module registered"}
        mock_requests.post.return_value = response
        
        result = self.agent.register_module(module_info)
        
        self.assertTrue(result)
        mock_requests.post.assert_called_once()

    def test_register_module_request_failure(self,
                                             mock_thread, mock_exists, 
                                             mock_check_deps, mock_requests):
        """Prueba manejo de falla en request de registro de módulo."""
        module_info = {"id": "test_module", "tipo": "auxiliar"}
        
        mock_requests.post.side_effect = requests.exceptions.RequestException()
        
        result = self.agent.register_module(module_info)
        
        self.assertFalse(result)

    def test_validate_auxiliary_module_ok(self,
                                          mock_thread, mock_exists, 
                                          mock_check_deps, mock_requests):
        """Prueba validación OK de módulo auxiliar completo."""
        module_info = {
            "id": "test_aux",
            "tipo": "auxiliar",
            "afinidad": "rendimiento",
            "naturaleza": "potenciador"
        }
        
        result = self.agent._validate_module(module_info)
        
        self.assertTrue(result)

    def test_validate_auxiliary_module_missing_afinidad(self,
                                                        mock_thread, 
                                                        mock_exists, 
                                                        mock_check_deps, 
                                                        mock_requests):
        """Prueba validación FAIL de auxiliar sin afinidad."""
        module_info = {
            "id": "test_aux",
            "tipo": "auxiliar",
            "naturaleza": "potenciador"
            # falta "afinidad"
        }
        
        result = self.agent._validate_module(module_info)
        
        self.assertFalse(result)

    def test_validate_auxiliary_module_missing_naturaleza(self,
                                                          mock_thread, 
                                                          mock_exists, 
                                                          mock_check_deps, 
                                                          mock_requests):
        """Prueba validación OK de auxiliar pero sin naturaleza 
        (no notifica)."""
        module_info = {
            "id": "test_aux",
            "tipo": "auxiliar",
            "afinidad": "estabilidad"
            # falta "naturaleza" pero es opcional para notificación
        }
        
        result = self.agent._validate_module(module_info)
        
        # Debería ser True porque afinidad está presente
        self.assertTrue(result)

    def test_validate_central_module_ok(self,
                                        mock_thread, mock_exists, 
                                        mock_check_deps, mock_requests):
        """Prueba validación OK de módulo central básico."""
        module_info = {
            "id": "test_central",
            "tipo": "central"
        }
        
        result = self.agent._validate_module(module_info)
        
        self.assertTrue(result)

    def test_validate_module_missing_id(self,
                                        mock_thread, mock_exists, 
                                        mock_check_deps, mock_requests):
        """Prueba validación FAIL sin ID."""
        module_info = {
            "tipo": "auxiliar",
            "afinidad": "estabilidad"
        }
        
        result = self.agent._validate_module(module_info)
        
        self.assertFalse(result)

    def test_validate_module_missing_tipo(self,
                                          mock_thread, mock_exists, 
                                          mock_check_deps, mock_requests):
        """Prueba validación FAIL sin tipo."""
        module_info = {
            "id": "test_module",
            "afinidad": "estabilidad"
        }
        
        result = self.agent._validate_module(module_info)
        
        self.assertFalse(result)

    def test_validate_module_invalid_tipo(self,
                                          mock_thread, mock_exists, 
                                          mock_check_deps, mock_requests):
        """Prueba validación FAIL con tipo inválido."""
        module_info = {
            "id": "test_module",
            "tipo": "tipo_invalido"
        }
        
        result = self.agent._validate_module(module_info)
        
        self.assertFalse(result)

    def test_validate_module_invalid_afinidad(self,
                                              mock_thread, mock_exists, 
                                              mock_check_deps, mock_requests):
        """Prueba validación FAIL con afinidad inválida."""
        module_info = {
            "id": "test_aux",
            "tipo": "auxiliar",
            "afinidad": "afinidad_invalida"
        }
        
        result = self.agent._validate_module(module_info)
        
        self.assertFalse(result)

    def test_validate_module_invalid_naturaleza(self,
                                                mock_thread, mock_exists, 
                                                mock_check_deps, 
                                                mock_requests):
        """Prueba validación FAIL con naturaleza inválida."""
        module_info = {
            "id": "test_aux",
            "tipo": "auxiliar",
            "afinidad": "estabilidad",
            "naturaleza": "naturaleza_invalida"
        }
        
        result = self.agent._validate_module(module_info)
        
        self.assertFalse(result)

    def test_calculate_pid_effort_empty_modules(self,
                                                mock_thread, mock_exists, 
                                                mock_check_deps, 
                                                mock_requests):
        """Prueba cálculo de esfuerzo PID con lista vacía."""
        modules = []
        
        effort = self.agent._calculate_pid_effort(modules)
        
        self.assertEqual(effort, 0.0)

    def test_calculate_pid_effort_single_module(self,
                                                mock_thread, mock_exists, 
                                                mock_check_deps, 
                                                mock_requests):
        """Prueba cálculo de esfuerzo PID con un módulo."""
        modules = [{"rendimiento": 0.8}]
        
        effort = self.agent._calculate_pid_effort(modules)
        
        # Esfuerzo = 1 - rendimiento_promedio = 1 - 0.8 = 0.2
        self.assertAlmostEqual(effort, 0.2, places=2)

    def test_calculate_pid_effort_multiple_modules(self,
                                                   mock_thread, mock_exists, 
                                                   mock_check_deps, 
                                                   mock_requests):
        """Prueba cálculo de esfuerzo PID con múltiples módulos."""
        modules = [
            {"rendimiento": 0.6},
            {"rendimiento": 0.8},
            {"rendimiento": 0.4}
        ]
        
        effort = self.agent._calculate_pid_effort(modules)
        
        # Promedio = (0.6 + 0.8 + 0.4) / 3 = 0.6
        # Esfuerzo = 1 - 0.6 = 0.4
        self.assertAlmostEqual(effort, 0.4, places=2)

    def test_calculate_pid_effort_missing_rendimiento(self,
                                                      mock_thread, 
                                                      mock_exists, 
                                                      mock_check_deps, 
                                                      mock_requests):
        """Prueba cálculo con módulo sin campo rendimiento."""
        modules = [
            {"rendimiento": 0.7},
            {"id": "mod_sin_rendimiento"}  # Sin campo rendimiento
        ]
        
        effort = self.agent._calculate_pid_effort(modules)
        
        # Debería usar solo los módulos con rendimiento válido
        self.assertAlmostEqual(effort, 0.3, places=2)  # 1 - 0.7


# Tests para métodos utilitarios adicionales
class TestAgentAIUtilities(unittest.TestCase):
    """Tests para métodos utilitarios de AgentAI."""

    def setUp(self):
        """Setup para tests de utilidades."""
        self.agent = AgentAI()

    def test_clamp_setpoint_within_bounds(self):
        """Prueba que setpoint se mantiene en límites válidos."""
        # Configurar límites de prueba
        min_val, max_val = -5.0, 5.0
        
        # Valor dentro de límites
        result = self.agent._clamp_setpoint(3.0, min_val, max_val)
        self.assertEqual(result, 3.0)

    def test_clamp_setpoint_above_max(self):
        """Prueba clamping de setpoint por encima del máximo."""
        min_val, max_val = -5.0, 5.0
        
        result = self.agent._clamp_setpoint(7.0, min_val, max_val)
        self.assertEqual(result, max_val)

    def test_clamp_setpoint_below_min(self):
        """Prueba clamping de setpoint por debajo del mínimo."""
        min_val, max_val = -5.0, 5.0
        
        result = self.agent._clamp_setpoint(-8.0, min_val, max_val)
        self.assertEqual(result, min_val)

    def test_normalize_signal_normal_range(self):
        """Prueba normalización de señal en rango normal."""
        signal = 0.5
        normalized = self.agent._normalize_signal(signal)
        
        self.assertGreaterEqual(normalized, 0.0)
        self.assertLessEqual(normalized, 1.0)
        self.assertAlmostEqual(normalized, 0.5, places=2)

    def test_normalize_signal_clamp_high(self):
        """Prueba normalización clampea valores altos."""
        signal = 1.5  # Por encima de 1.0
        normalized = self.agent._normalize_signal(signal)
        
        self.assertEqual(normalized, 1.0)

    def test_normalize_signal_clamp_low(self):
        """Prueba normalización clampea valores bajos."""
        signal = -0.5  # Por debajo de 0.0
        normalized = self.agent._normalize_signal(signal)
        
        self.assertEqual(normalized, 0.0)


# Ejecutar tests
if __name__ == "__main__":
    # Configurar para tests más verbosos si es necesario
    # runner = unittest.TextTestRunner(verbosity=2)
    # unittest.main(testRunner=runner)
    unittest.main()

# --- END OF FILE tests/unit/test_agent_ai.py (AJUSTADO Fase 1) ---
