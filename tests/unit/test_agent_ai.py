#!/usr/bin/env python3
"""Pruebas unitarias para el módulo `agent_ai.agent_ai`.

Este módulo contiene la suite de pruebas unitarias para la clase `AgentAI`,
que es el núcleo estratégico encargado de la toma de decisiones y la
interacción con otros componentes del sistema como Harmony Controller y
los módulos auxiliares. Las pruebas cubren la inicialización, la comunicación,
la lógica de decisión basada en estrategias y la gestión de módulos.
"""

import unittest
import unittest.mock as mock
import os
import numpy as np
import requests

from agent_ai.agent_ai import (
    AgentAI,
    HARMONY_CONTROLLER_URL_ENV,
    HARMONY_CONTROLLER_REGISTER_URL_ENV,
    AGENT_AI_ECU_URL_ENV,
    AGENT_AI_MALLA_URL_ENV,
    DEFAULT_HC_URL,
    DEFAULT_ECU_URL,
    DEFAULT_MALLA_URL,
    GLOBAL_REQUIREMENTS_PATH,
    REQUESTS_TIMEOUT,
    MAX_RETRIES
)


@mock.patch("agent_ai.agent_ai.requests", new_callable=mock.MagicMock)
@mock.patch("agent_ai.agent_ai.check_missing_dependencies")
@mock.patch("agent_ai.agent_ai.validate_module_registration")
@mock.patch("agent_ai.agent_ai.logger")
class TestAgentAI(unittest.TestCase):
    """
    Conjunto de pruebas unitarias para la clase `AgentAI`.

    Esta clase de pruebas verifica el comportamiento de `AgentAI`,
    incluyendo la inicialización con valores por defecto y variables
    de entorno, la comunicación con Harmony Controller (obtención de
    estado y envío de setpoints), la lógica interna para determinar
    el setpoint de Harmony basado en diferentes estrategias y señales
    externas, el registro y validación de módulos auxiliares y centrales,
    y la gestión de comandos estratégicos y la obtención del estado
    completo del agente.

    Attributes:
        agent (AgentAI): Instancia de `AgentAI` creada para cada prueba.
        mock_sleep Mock para `time.sleep` usado en reintentos.
    """

    def setUp(self):
        """
        Configura el entorno necesario antes de la ejecución de cada prueba.

        Este método se invoca automáticamente por `unittest` antes de cada
        método de prueba. Se encarga de limpiar variables de entorno relevantes
        que podrían afectar la inicialización de `AgentAI` y crea una nueva
        instancia de `AgentAI`. También se asegura de que si un hilo
        estratégico de una prueba anterior sigue vivo, este sea detenido.
        Además, inicializa un mock para `time.sleep` que se usa para
        controlar los reintentos en las comunicaciones.
        """
        os.environ.pop(HARMONY_CONTROLLER_URL_ENV, None)
        os.environ.pop(HARMONY_CONTROLLER_REGISTER_URL_ENV, None)
        os.environ.pop(AGENT_AI_ECU_URL_ENV, None)
        os.environ.pop(AGENT_AI_MALLA_URL_ENV, None)
        os.environ.pop("AA_INITIAL_SETPOINT_VECTOR", None)
        os.environ.pop("AA_INITIAL_STRATEGY", None)

        self.agent = AgentAI()
        if self.agent._strategic_thread.is_alive():
            self.agent.shutdown()

        # Mock time.sleep
        self.patcher_sleep = mock.patch("agent_ai.agent_ai.time.sleep")
        self.mock_sleep = self.patcher_sleep.start()

    def tearDown(self):
        """
        Limpia el entorno después de la ejecución de cada prueba.

        Este método se invoca automáticamente por `unittest` después de
        cada método de prueba. Su principal responsabilidad es asegurar
        que el hilo estratégico (`_strategic_thread`) del agente `AgentAI`
        sea detenido correctamente si aún está en ejecución, para evitar
        interferencias entre pruebas.
        """
        if self.agent._strategic_thread.is_alive():
            self.agent.shutdown()

        # Stop mock_sleep patcher
        self.patcher_sleep.stop()

    def test_initialization_defaults(
        self, mock_get_logger, mock_validate, mock_check_deps, mock_requests
    ):
        """
        Verifica la inicialización de AgentAI con valores por defecto.

        Esta prueba asegura que cuando `AgentAI` se instancia sin variables
        de entorno específicas que modifiquen su configuración inicial,
        se establecen correctamente los valores predeterminados para sus
        atributos, tales como la lista de módulos (vacía), la estrategia
        actual ('default'), el vector de setpoint objetivo, y las URLs
        para los servicios centrales (Harmony Controller, ECU, Malla Watcher).

        Args:
            mock_get_logger, Mock del logger.
            mock_validate, Mock de la función de validación de registro.
            mock_check_deps, Mock de la función de chequeo de dependencias.
            mock_requests, Mock del módulo `requests`.
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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Verifica que AgentAI usa URLs por defecto si no hay variables
        de entorno.

        Esta prueba comprueba que, en ausencia de variables de entorno
        que especifiquen las URLs para Harmony Controller, ECU y
        Malla Watcher, `AgentAI` utiliza correctamente las URLs
        predeterminadas definidas internamente. También verifica que la
        URL de registro en Harmony Controller se construya a partir de la
        URL base por defecto de HC.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Verifica que AgentAI usa URLs de variables de entorno
        cuando están definidas.

        Esta prueba asegura que si las variables de entorno para
        las URLs de Harmony Controller, ECU, Malla Watcher y el
        endpoint de registro de HC están definidas, `AgentAI`
        prioriza estos valores sobre los predeterminados durante
        su inicialización.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Prueba la obtención exitosa del estado de Harmony Controller.

        Verifica que el método `_get_harmony_state` puede procesar
        correctamente una respuesta exitosa (código 200) del endpoint
        de estado de Harmony Controller. Asegura que la URL correcta es
        llamada y que los datos del estado son
        extraídos y devueltos adecuadamente.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Prueba el manejo de errores de red al obtener el estado de Harmony.

        Verifica que `_get_harmony_state` maneja adecuadamente los errores
        de red (simulados por `requests.exceptions.RequestException`).
        Se espera que el método intente la comunicación `MAX_RETRIES`
        veces, con pausas entre intentos (verificadas por `mock_sleep`),
        y que finalmente devuelva `None` si el error persiste.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Prueba el manejo de respuestas no exitosas de Harmony Controller.

        Verifica que `_get_harmony_state` devuelve `None` cuando Harmony
        Controller responde con un JSON que indica un error
        (ej. status 'error'), incluso si el código de estado HTTP es 200.
        La prueba se configura para un solo intento (MAX_RETRIES = 1)
        para aislar este comportamiento.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Prueba el envío exitoso de un setpoint a Harmony Controller.

        Verifica que el método `_send_setpoint_to_harmony` construye
        correctamente la solicitud POST (URL y payload JSON) para
        enviar un nuevo vector de setpoint a Harmony Controller y que
        maneja adecuadamente una respuesta exitosa (código 200).

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Prueba el manejo de errores de red al enviar setpoint a Harmony.

        Verifica que `_send_setpoint_to_harmony` maneja errores de red
        (simulados por `requests.exceptions.RequestException`) durante
        el envío de un setpoint. Se espera que el método reintente la
        operación `MAX_RETRIES` veces, con pausas intermedias, y que no
        levante una excepción si el error persiste
        (la gestión de errores es interna).

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
        mock_requests.post.side_effect = requests.exceptions.RequestException(
            "Connection failed"
        )
        setpoint_vec = [1.5, -0.5]
        self.agent._send_setpoint_to_harmony(setpoint_vec)
        # Verificar que reintentó MAX_RETRIES veces
        self.assertEqual(mock_requests.post.call_count, MAX_RETRIES)
        self.assertEqual(self.mock_sleep.call_count, MAX_RETRIES - 1)

    def test_determine_harmony_setpoint_simple(
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Prueba la lógica de determinación del setpoint de Harmony
        en varios escenarios.

        Este método de prueba cubre múltiples casos para la función
        `_determine_harmony_setpoint`:
        - Estrategia 'default' sin factores externos: el setpoint no cambia.
        - Señal de Cogniboard alta: el setpoint se reduce proporcionalmente.
        - Señal de Cogniboard inválida: el setpoint no cambia.
        - Estrategia 'estabilidad':
            - Con error bajo o esfuerzo PID alto: la magnitud del setpoint
            se reduce.
            - Con más módulos reductores que potenciadores: se aplica una
            reducción adicional.
        - Estrategia 'rendimiento':
            - Con error bajo y esfuerzo PID bajo: la magnitud del setpoint
            aumenta.
            - Con más módulos potenciadores (ECU) que reductores: se aplica
            un aumento adicional.
        - Estrategia 'ahorro_energia':
            - Con módulos reductores activos:
            la magnitud del setpoint se reduce.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """

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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Verifica que la estrategia 'default' no altera el setpoint base.

        Cuando la estrategia es 'default' y no hay otras influencias
        (como señales de Cogniboard o módulos auxiliares específicos
        que modifiquen el comportamiento base), el setpoint determinado
        debe ser igual al setpoint objetivo actual del agente.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self, mock_thread, mock_validate_registration, mock_check_deps, mock_requests
    ):
        """
        Verifica que la estrategia 'estabilidad' reduce la magnitud del
        setpoint si el error es bajo.

        Si la estrategia activa es 'estabilidad' y la medición actual
        está muy cerca del setpoint objetivo (error bajo), y el esfuerzo
        del PID es bajo, se espera que la magnitud del nuevo setpoint se
        reduzca en un factor predeterminado (0.98).

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica que 'estabilidad' reduce magnitud del setpoint si
        el esfuerzo PID es alto.

        Con la estrategia 'estabilidad', si el último esfuerzo registrado
        del controlador PID es alto (indicando dificultad para alcanzar
        el setpoint), la magnitud del nuevo setpoint debe reducirse en
        un factor predeterminado (0.98), independientemente del error actual.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica reducción extra en 'estabilidad' si hay más reductores
        que potenciadores en malla.

        Si la estrategia es 'estabilidad', y además de una condición de
        reducción base (como error bajo), hay más módulos auxiliares de
        tipo 'reductor' que 'potenciador' afectando a 'malla_watcher', se
        debe aplicar una reducción adicional a la magnitud del setpoint
        (factor 0.97 sobre la reducción anterior).

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica que la estrategia 'rendimiento' aumenta la magnitud
        del setpoint si el sistema está estable.

        Cuando la estrategia es 'rendimiento', el error es bajo y el
        esfuerzo del PID es bajo (indicando estabilidad y capacidad
        de respuesta), se espera que la magnitud del setpoint aumente
        en un factor predeterminado (1.02).

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
            "Rendimiento/Estable: Norma debe aumentar.",
        )
        np.testing.assert_allclose(final_norm, initial_norm * 1.02, rtol=1e-6)

    def test_determine_estrategia_rendimiento_aumenta_extra_por_potenciadores(
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica aumento extra en 'rendimiento' si hay más
        potenciadores que reductores en ECU.

        Si la estrategia es 'rendimiento', el sistema está estable,
        y además hay más módulos auxiliares de tipo 'potenciador'
        que 'reductor' afectando a 'matriz_ecu', se debe aplicar
        un aumento adicional a la magnitud del setpoint
        (factor 1.01 sobre el aumento anterior).

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
            measurement,
            None,
            None,
            "rendimiento",
            modules
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica que 'rendimiento' establece un valor mínimo si el
        setpoint actual es cero.

        Si la estrategia es 'rendimiento' y el setpoint objetivo
        actual es cero (o un vector nulo), y el sistema está estable,
        el nuevo setpoint no debe permanecer en cero. En su lugar, se
        debe establecer un valor mínimo predeterminado
        (ej. [0.1, 0.1] para un vector 2D) para iniciar la actividad.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica que 'ahorro_energia' reduce magnitud si hay
        módulos reductores activos.

        Con la estrategia 'ahorro_energia', si existen módulos
        auxiliares activos de tipo 'reductor' (independientemente
        de a qué componente afecten), la magnitud del setpoint
        debe reducirse en un factor predeterminado (0.95).

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica que 'ahorro_energia' no cambia el setpoint
        si no hay reductores.

        Si la estrategia es 'ahorro_energia' pero no hay módulos
        auxiliares de tipo 'reductor' activos (aunque haya otros
        tipos como 'potenciador'), el setpoint determinado no debe
        cambiar respecto al setpoint objetivo actual del agente.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica que una señal alta de Cogniboard reduce la
        magnitud del setpoint.

        Independientemente de la estrategia, si se recibe una
        señal numérica válida de Cogniboard (entre 0 y 1), esta
        actúa como un factor de escala directo sobre la magnitud
        del setpoint calculado por la estrategia. Una señal de 0.9
        debería reducir la magnitud a un 90% de la original.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
    @mock.patch("agent_ai.agent_ai.threading.Thread")
    def test_registrar_modulo_auxiliar_success(
        self,
        mock_threading_thread,
        mock_logger,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Prueba el registro exitoso de un módulo auxiliar con
        todos sus atributos.

        Verifica que el método `registrar_modulo` procesa
        correctamente los datos de un módulo de tipo 'auxiliar',
        incluyendo su nombre, URLs, tipo, a qué componente aporta
        (`aporta_a`), su naturaleza (`naturaleza_auxiliar`),
        y la ruta a sus requisitos. Asegura que el módulo se añade
        al diccionario interno `agent.modules` con el estado 'pendiente',
        y que se intenta iniciar un hilo para su validación de salud.
        También comprueba que `check_missing_dependencies` es llamado.

        Args:
            mock_threading_thread, Mock para threading.Thread.
            mock_logger, Mock para el logger.
            mock_validate_registration, Mock para `validate_module_registration`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
        mock_check_deps.return_value = (True, "Dependencias OK")
        mock_validate_registration.return_value = (True, "Mock validation successful")

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
        mock_threading_thread.assert_called_once()
        # Verificar que check_missing_dependencies fue llamado
        mock_check_deps.assert_called_once_with(
            "dummy_req.txt", GLOBAL_REQUIREMENTS_PATH
        )

    @mock.patch("agent_ai.agent_ai.threading.Thread")
    def test_registrar_modulo_central_success(
        self,
        mock_threading_thread,
        mock_logger,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Prueba el registro exitoso de un módulo central.

        Verifica que `registrar_modulo` puede registrar un
        módulo de tipo 'central' (o 'integrador', etc.), que
        no requiere los campos `aporta_a` ni `naturaleza_auxiliar`.
        Asegura que el módulo se añade correctamente a `agent.modules`
        y que se inicia el hilo de validación. No se espera que se llame
        a `check_missing_dependencies` si no se proporciona
        `requirements_path`.

        Args:
            mock_threading_thread, Mock para threading.Thread.
            mock_logger, Mock para el logger.
            mock_validate_registration, Mock para `validate_module_registration`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
        mock_check_deps.return_value = (True, "Dependencias OK")
        mock_validate_registration.return_value = (True, "Mock validation successful")
        module_data = {
            "nombre": "CentralTest",
            "url": "http://centraltest:5678/api/state",
            "url_salud": "http://centraltest:5678/api/health",
            "tipo": "central",  # O integrador
        }
        result = self.agent.registrar_modulo(module_data)

        self.assertEqual(result["status"], "success")
        self.assertIn("CentralTest", self.agent.modules)
        module_entry = self.agent.modules["CentralTest"]
        self.assertEqual(module_entry["tipo"], "central")
        self.assertNotIn("aporta_a", module_entry)
        self.assertNotIn("naturaleza_auxiliar", module_entry)
        mock_threading_thread.assert_called_once()

    def test_registrar_modulo_invalid_data(
        self,
        mock_logger,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Prueba el intento de registro de un módulo con datos
        inválidos o faltantes.

        Verifica que `registrar_modulo` devuelve un error si
        los datos proporcionados para el registro del módulo
        son insuficientes o inválidos, según la validación
        realizada por `validate_module_registration`.
        El módulo no debe ser añadido a `agent.modules` en este caso.

        Args:
            mock_logger, Mock para el logger.
            mock_validate_registration, Mock para `validate_module_registration`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
        # Asumiendo que validator.validate_module_registration verifica campos
        # requeridos
        module_data = {"nombre": "TestReg"}  # Falta url, tipo, etc.
        # Mockear el validador para simular fallo
        mock_validate_registration.return_value = (
            False,
            "Faltan campos requeridos: url, tipo"
        )
        result = self.agent.registrar_modulo(module_data)
        self.assertEqual(result["status"], "error")
        self.assertIn("Faltan campos requeridos", result["mensaje"])
        self.assertNotIn("TestReg", self.agent.modules)

    def test_registrar_modulo_dep_fail(
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Prueba el intento de registro de un módulo cuando falla la
        verificación de dependencias.

        Verifica que si `check_missing_dependencies`
        (llamado durante el registro de un módulo que especifica un
        `requirements_path`) devuelve que faltan dependencias, el
        método `registrar_modulo` devuelve un error y el módulo no
        se añade a `agent.modules`.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
        mock_check_deps.return_value = (False, "Falta 'superlib'")
        mock_validate_registration.return_value = True
        # For this test, we want validate_module_registration to succeed,
        # so the dependency check is the point of failure.
        mock_validate_registration.return_value = (True, "Mock validation successful")
        module_data = {
            "nombre": "TestDepFail",
            "url": "http://testdep/health",
            "tipo": "auxiliar",
            "aporta_a": "malla_watcher",  # Added to pass validation
            "naturaleza_auxiliar": "reductor",  # Added to pass validation
            "requirements_path": "req.txt",
        }
        result = self.agent.registrar_modulo(module_data)
        self.assertEqual(result["status"], "error")
        self.assertIn("Falta 'superlib'", result["mensaje"])
        self.assertNotIn("TestDepFail", self.agent.modules)

    def test_validar_salud_modulo_ok_auxiliar_sin_naturaleza(
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Prueba la validación de salud exitosa de un módulo auxiliar
        sin naturaleza definida.

        Verifica que `_validar_salud_modulo` actualiza el estado de
        salud de un módulo a 'ok' si la petición a su `url_salud` es
        exitosa (código 200). Si el módulo es de tipo 'auxiliar' pero
        no tiene `naturaleza_auxiliar` definida, no se debe intentar
        notificar a Harmony Controller.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Prueba la validación de salud exitosa de un módulo de tipo
        no-auxiliar (ej. 'central').

        Verifica que `_validar_salud_modulo` actualiza el estado de
        salud a 'ok' para módulos que no son 'auxiliar'
        (ej. 'central', 'integrador') si la respuesta de su `url_salud`
        es exitosa. En estos casos, no se debe intentar notificar a
        Harmony Controller, independientemente de otros atributos.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """Prueba la validación de salud fallida de un módulo.

        Verifica que si la petición a la `url_salud` de un módulo falla
        (ej. por `requests.exceptions.ConnectionError`) después de los
        reintentos configurados, el estado de salud del módulo se
        actualiza a 'error_inesperado'. Crucialmente, en caso de fallo
        de validación de salud, no se debe intentar notificar a
        Harmony Controller, incluso si el módulo es auxiliar
        y tiene naturaleza definida.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Prueba el mecanismo de reintentos al notificar a Harmony Controller.

        Verifica que la función intenta realizar la notificación
        (una solicitud POST) a Harmony Controller si la solicitud falla
        persistentemente (ej. por `requests.exceptions.RequestException`).
        También asegura que se realizan pausas (`mock_sleep`) entre
        los intentos.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """
        Prueba la funcionalidad de actualizar comandos estratégicos del agente.

        Verifica la correcta gestión de diferentes comandos a través de
        `actualizar_comando_estrategico`:
        - `set_strategy`: Cambia la `current_strategy` del agente.
        - `set_target_setpoint_vector`: Actualiza el `target_setpoint_vector` y
          envía inmediatamente el nuevo setpoint a Harmony Controller.
        - Comando inválido: Devuelve un error apropiado.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
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
        self,
        mock_thread,
        mock_validate_registration,
        mock_check_deps,
        mock_requests
    ):
        """Prueba la recepción y almacenamiento de inputs externos.

        Verifica que los métodos `recibir_control_cogniboard` y
        `recibir_config_status` actualizan correctamente el diccionario
        `external_inputs` del agente con las señales y datos recibidos.

        Args:
            mock_thread, Mock para la creación de hilos.
            mock_os_exists, Mock para `os.path.exists`.
            mock_check_deps, Mock para `check_missing_dependencies`.
            mock_requests, Mock para el módulo `requests`.
        """
        self.agent.recibir_control_cogniboard(0.77)
        self.assertEqual(self.agent.external_inputs["cogniboard_signal"], 0.77)
        config_data = {"status": "healthy"}
        self.agent.recibir_config_status(config_data)
        self.assertEqual(
            self.agent.external_inputs["config_status"], config_data
        )

    # --- MODIFICADO: test_obtener_estado_completo incluye naturaleza ---
    def test_get_full_state_snapshot(
        self,
        mock_get_logger,
        mock_validate,
        mock_check_deps,
        mock_requests
    ):
        """
        Verifica que `obtener_estado_completo` retorna una instantánea
        precisa del estado del agente.

        Esta prueba configura diversos aspectos del estado interno del
        agente `AgentAI` (setpoint objetivo, estrategia actual,
        inputs externos, estado de Harmony, módulos registrados
        con diferentes características) y luego llama a
        `obtener_estado_completo`. Comprueba que el diccionario devuelto
        refleja fielmente todos estos aspectos, incluyendo los detalles
        de los módulos como tipo, `aporta_a` y `naturaleza_auxiliar`
        cuando corresponda.

        Args:
            mock_get_logger, Mock del logger.
            mock_validate, Mock de la función de validación de registro.
            mock_check_deps, Mock de la función de chequeo de dependencias.
            mock_requests, Mock del módulo `requests`.
        """
        with self.agent.lock:
            self.agent.target_setpoint_vector = [0.5, -0.5]
            self.agent.current_strategy = "test_strat"
            self.agent.external_inputs["cogniboard_signal"] = 0.1
            self.agent.harmony_state = {"last_measurement": 0.9}
            self.agent.modules = {
                "TestModCentral": {
                    "nombre": "TestModCentral",
                    "tipo": "integrador",
                    "estado_salud": "error_timeout",
                },
                "TestModAux": {
                    "nombre": "TestModAux",
                    "tipo": "auxiliar",
                    "aporta_a": "malla_watcher",
                    "naturaleza_auxiliar": "modulador",
                    "estado_salud": "ok",
                },
            }

        estado = self.agent.obtener_estado_completo()

        self.assertListEqual(estado["target_setpoint_vector"], [0.5, -0.5])
        self.assertEqual(estado["current_strategy"], "test_strat")
        self.assertEqual(
            estado["external_inputs"]["cogniboard_signal"], 0.1
        )
        self.assertEqual(
            estado["harmony_controller_last_state"]["last_measurement"],
            0.9,
        )
        self.assertEqual(len(estado["registered_modules"]), 2)

        mod_aux = next(
            m
            for m in estado["registered_modules"]
            if m["nombre"] == "TestModAux"
        )
        self.assertEqual(mod_aux["tipo"], "auxiliar")
        self.assertEqual(mod_aux["aporta_a"], "malla_watcher")
        self.assertEqual(mod_aux["naturaleza_auxiliar"], "modulador")
        self.assertEqual(mod_aux["estado_salud"], "ok")

        mod_central = next(
            m
            for m in estado["registered_modules"]
            if m["nombre"] == "TestModCentral"
        )
        self.assertEqual(mod_central["tipo"], "integrador")
        self.assertNotIn("aporta_a", mod_central)
        self.assertNotIn("naturaleza_auxiliar", mod_central)
        self.assertEqual(mod_central["estado_salud"], "error_timeout")


if __name__ == "__main__":
    unittest.main()
