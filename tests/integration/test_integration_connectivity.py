# --- START OF FILE tests/integration/test_integration_connectivity.py ---

import pytest
import requests
import os
import time  # Necesario para time.sleep
import logging
import json

# Configurar logging para pruebas de integración
# Esto ayuda a ver qué URLs se están intentando y las respuestas
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# --- Configuración: URLs de Servicios ---
# Leer URLs desde variables de entorno, proporcionar valores por defecto para
# pruebas locales. Los valores por defecto deben coincidir con los puertos
# expuestos por tu docker-compose.yml. Asegúrate de que estos puertos son
# accesibles desde donde ejecutas pytest.
ECU_URL = os.environ.get("TEST_ECU_URL", "http://localhost:8000")
MALLA_URL = os.environ.get("TEST_MALLA_URL", "http://localhost:5001")
HARMONY_CONTROLLER_URL = os.environ.get(
    "TEST_HC_URL", "http://localhost:7000"
)
AGENT_AI_URL = os.environ.get("TEST_AGENT_AI_URL", "http://localhost:9000")

# Configuración para reintentos en health checks
HEALTH_CHECK_TIMEOUT = 5  # Timeout en segundos para cada solicitud
HEALTH_CHECK_RETRIES = 10  # Número de reintentos para el health check
RETRY_DELAY = 3  # Tiempo de espera en segundos entre reintentos


# --- Función auxiliar para health checks con reintentos ---
def check_service_health(service_name, url):
    """
    Intenta conectar a la URL del servicio y verifica el estado de salud
    con reintentos. Retorna (True/False, data) donde data es la
    respuesta JSON.
    """
    logger.info(f"Verificando salud de {service_name} en {url}")
    attempts = 0
    data = {}  # Inicializar data para el caso de fallo total

    while attempts < HEALTH_CHECK_RETRIES:
        try:
            # Ajustar el endpoint según el servicio
            state_endpoint = url
            if "ecu" in url.lower():
                state_endpoint += "/api/ecu/state"
            elif "malla" in url.lower():
                state_endpoint += "/api/malla/state"
            # Acepta hc o harmony_controller en la URL para el endpoint de HC
            elif "harmony_controller" in url.lower() or \
                 "hc" in url.lower():
                state_endpoint += "/api/harmony/state"
            elif "agent_ai" in url.lower():
                state_endpoint += "/api/state"
            else:
                # Endpoint por defecto si no se reconoce el nombre en la URL
                state_endpoint += "/api/state"

            response = requests.get(state_endpoint,
                                    timeout=HEALTH_CHECK_TIMEOUT)
            # Lanza excepción para códigos de status 4xx/5xx
            response.raise_for_status()
            data = response.json()

            # Verifica si el servicio reporta estar saludable
            if data.get("is_healthy", False):
                logger.info(
                    f"{service_name} saludable. Estado reportado: {data}"
                )
                return True, data
            else:
                # Si responde 200 pero is_healthy es False, reintentar
                logger.warning(
                    f"{service_name} accesible pero reporta is_healthy: "
                    f"False. Reintentando..."
                )

        except requests.exceptions.Timeout:
            logger.warning(
                f"Timeout ({HEALTH_CHECK_TIMEOUT}s) conectando con "
                f"{service_name} en {state_endpoint}. Intento "
                f"{attempts + 1}/{HEALTH_CHECK_RETRIES}"
            )
        except requests.exceptions.ConnectionError as e:
            logger.warning(
                f"Error de conexión con {service_name} en {state_endpoint}: "
                f"{e}. Intento {attempts + 1}/{HEALTH_CHECK_RETRIES}"
            )
        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Error HTTP general con {service_name} en {state_endpoint}: "
                f"{e}. Intento {attempts + 1}/{HEALTH_CHECK_RETRIES}"
            )
        except json.JSONDecodeError:
            # Guardar response.text si existe para el log
            response_text = response.text if 'response' in locals() else "N/A"
            logger.warning(
                f"Respuesta de {service_name} no es JSON válido. URL: "
                f"{state_endpoint}, Respuesta: {response_text}. Intento "
                f"{attempts + 1}/{HEALTH_CHECK_RETRIES}"
            )
        except Exception:  # F841: e is implicitly used by logger.exception
            logger.exception(
                f"Excepción inesperada verificando {service_name} en "
                f"{state_endpoint}. Intento {attempts + 1}/"
                f"{HEALTH_CHECK_RETRIES}"
            )

        attempts += 1
        if attempts < HEALTH_CHECK_RETRIES:
            time.sleep(RETRY_DELAY)  # Esperar antes de reintentar

    logger.error(
        f"FALLO: {service_name} no respondió saludablemente después de "
        f"{HEALTH_CHECK_RETRIES} reintentos."
    )
    return False, data if data else {"error": "No data received"}


# --- Tests de Conectividad ---

# Opcional: Fixture para una espera inicial antes de todos los tests
# @pytest.fixture(scope="module", autouse=True)
# def initial_wait_for_services():
# """Espera un tiempo al inicio del módulo para dar tiempo a los servicios
# a arrancar."""
#     wait_time = int(os.environ.get("SERVICE_STARTUP_WAIT", 15))
#     logger.info(
#         f"Esperando {wait_time} segundos para que servicios se inicien..."
#     )
#     time.sleep(wait_time)
#     logger.info("Espera inicial finalizada. Iniciando health checks.")


@pytest.mark.real_integration
def test_ecu_is_accessible_and_healthy():
    """
    Verificar que Matriz ECU es accesible, reporta salud y su estado
    inicial (la matriz) tiene la estructura esperada.
    """
    is_healthy, data = check_service_health("Matriz ECU", ECU_URL)

    assert is_healthy, (
        "Matriz ECU no saludable/accesible tras reintentos. "
        "Última resp.:\n"
        f"    {data}"
    )

    logger.info("Verificando estructura del estado de Matriz ECU...")

    assert "data" in data, (
        f"Respuesta de {ECU_URL}/api/ecu/state no contiene 'data'. "
        f"Respuesta: {data}"
    )
    ecu_state_data = data["data"]

    assert "matrix_state" in ecu_state_data, (
        f"'matrix_state' no en datos de Matriz ECU. Datos: {ecu_state_data}"
    )
    matrix_list = ecu_state_data["matrix_state"]

    assert isinstance(matrix_list, list), (
        f"matrix_state no es una lista. Tipo: {type(matrix_list)}. "
        f"Datos: {ecu_state_data}"
    )

    # Dimensiones esperadas (deben coincidir con matriz_ecu.py)
    expected_capas = 3
    expected_filas = 10
    expected_columnas = 20
    expected_vector_dim = 2  # [vx, vy]

    assert len(matrix_list) == expected_capas, (
        f"Número incorrecto de capas. Esperado: {expected_capas}, "
        f"Obtenido: {len(matrix_list)}. Datos: {ecu_state_data}"
    )

    if expected_capas > 0:
        assert isinstance(matrix_list[0], list), (
            f"Capas no contienen listas (filas). Tipo: "
            f"{type(matrix_list[0])}. Datos: {ecu_state_data}"
        )
        assert len(matrix_list[0]) == expected_filas, (
            f"Número incorrecto de filas. Esperado: {expected_filas}, "
            f"Obtenido: {len(matrix_list[0])}. Datos: {ecu_state_data}"
        )

        if expected_filas > 0:
            assert isinstance(matrix_list[0][0], list), (
                f"Filas no contienen listas (columnas). Tipo: "
                f"{type(matrix_list[0][0])}. Datos: {ecu_state_data}"
            )
            assert len(matrix_list[0][0]) == expected_columnas, (
                f"Número incorrecto de columnas. Esperado: "
                f"{expected_columnas}, Obtenido: {len(matrix_list[0][0])}. "
                f"Datos: {ecu_state_data}"
            )

            if expected_columnas > 0:
                assert isinstance(matrix_list[0][0][0], list), (
                    f"Columnas no contienen listas (vectores). Tipo: "
                    f"{type(matrix_list[0][0][0])}. Datos: {ecu_state_data}"
                )
                assert len(matrix_list[0][0][0]) == expected_vector_dim, (
                    f"Dimensión incorrecta del vector [vx, vy]. Esperado: "
                    f"{expected_vector_dim}, Obtenido: "
                    f"{len(matrix_list[0][0][0])}. Datos: {ecu_state_data}"
                )
                assert all(
                    isinstance(val, (int, float))
                    for val in matrix_list[0][0][0]
                ), (
                    f"Elementos del vector [vx, vy] no son números. "
                    f"Vector: {matrix_list[0][0][0]}. Datos: {ecu_state_data}"
                )

    if expected_capas > 0 and expected_filas > 0 and expected_columnas > 0:
        assert abs(matrix_list[0][0][0][0]) < 1e-9, (
            f"Componente X del vector [0,0,0] no es cero. "
            f"Valor: {matrix_list[0][0][0][0]}. Datos: {ecu_state_data}"
        )
        assert abs(matrix_list[0][0][0][1]) < 1e-9, (
            f"Componente Y del vector [0,0,0] no es cero. "
            f"Valor: {matrix_list[0][0][0][1]}. Datos: {ecu_state_data}"
        )

    logger.info(
        "Verificación de estructura y estado inicial de Matriz ECU PASADA."
    )


@pytest.mark.real_integration
def test_malla_watcher_is_accessible_and_healthy():
    """Verificar que Malla Watcher es accesible y reporta salud."""
    is_healthy, data = check_service_health("Malla Watcher", MALLA_URL)
    assert is_healthy, (
        f"Malla Watcher no saludable/accesible. Última resp: {data}"
    )
    # Opcional: Aserciones específicas sobre Malla
    # assert data.get("details", {}).get("mesh", {}) \
    # .get("initialized") is True, "La malla no está inicializada"
    # assert data.get("details", {}).get("resonator_simulation", {}) \
    # .get("running") is True, "Simulación de Malla no corre"
    # assert data.get("details", {}).get("mesh", {}) \
    # .get("num_cells", 0) > 0, "La malla está vacía"


@pytest.mark.real_integration
def test_harmony_controller_is_accessible_and_healthy():
    """Verificar que Harmony Controller es accesible y reporta salud."""
    is_healthy, data = check_service_health(
        "Harmony Controller", HARMONY_CONTROLLER_URL
    )
    assert is_healthy, (
        f"Harmony Controller no saludable/accesible. Última resp: {data}"
    )
    # Opcional: Aserciones específicas sobre HC
    # assert data.get("control_loop_running") is True, \
    # "Bucle de control de HC no corre"


@pytest.mark.real_integration
def test_agent_ai_is_accessible_and_healthy():
    """Verificar que Agent AI es accesible y reporta salud."""
    is_healthy, data = check_service_health("Agent AI", AGENT_AI_URL)
    assert is_healthy, (
        f"Agent AI no saludable/accesible. Última resp: {data}"
    )
    # Opcional: Aserciones específicas sobre AgentAI
    # assert data.get("status") == "success", \
    # "Agent AI no reporta estado success"

# --- END OF FILE tests/integration/test_integration_connectivity.py ---
