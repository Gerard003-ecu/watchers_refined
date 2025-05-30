# --- START OF FILE tests/integration/test_integration_connectivity.py ---

import pytest
import requests
import os
import time # Necesario para time.sleep
import logging
from typing import Optional

# Configurar logging para pruebas de integración
# Esto ayuda a ver qué URLs se están intentando y las respuestas
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Configuración: URLs de Servicios ---
# Leer URLs desde variables de entorno, proporcionar valores por defecto para pruebas locales
# Los valores por defecto deben coincidir con los puertos expuestos por tu docker-compose.yml
# Asegúrate de que estos puertos son accesibles desde donde ejecutas pytest
ECU_URL = os.environ.get("TEST_ECU_URL", "http://localhost:8000")
MALLA_URL = os.environ.get("TEST_MALLA_URL", "http://localhost:5001")
HARMONY_CONTROLLER_URL = os.environ.get("TEST_HC_URL", "http://localhost:7000")
AGENT_AI_URL = os.environ.get("TEST_AGENT_AI_URL", "http://localhost:9000")

# Configuración para reintentos en health checks
HEALTH_CHECK_TIMEOUT = 5 # Timeout para cada solicitud individual (en segundos)
HEALTH_CHECK_RETRIES = 10 # Número de veces que se intentará el health check
RETRY_DELAY = 3 # Tiempo de espera entre reintentos (en segundos)

# --- Función auxiliar para health checks con reintentos ---
def check_service_health(service_name, url):
    """
    Intenta conectar a la URL del servicio y verifica el estado de salud
    con reintentos. Retorna (True/False, data) donde data es la respuesta JSON.
    """
    logger.info(f"Verificando salud de {service_name} en {url}")
    attempts = 0
    while attempts < HEALTH_CHECK_RETRIES:
        try:
            # Ajustar el endpoint según el servicio, si no todos tienen /api/state
            # ECU usa /api/ecu/state, Malla usa /api/malla/state, HC usa /api/harmony/state, AgentAI usa /api/state
            state_endpoint = url
            if "ecu" in url.lower():
                 state_endpoint += "/api/ecu/state"
            elif "malla" in url.lower():
                 state_endpoint += "/api/malla/state"
            elif "harmony_controller" in url.lower() or "hc" in url.lower(): # Acepta hc o harmony_controller
                 state_endpoint += "/api/harmony/state"
            elif "agent_ai" in url.lower():
                 state_endpoint += "/api/state"
            else:
                 # Endpoint por defecto si no se reconoce el nombre en la URL
                 state_endpoint += "/api/state"


            response = requests.get(state_endpoint, timeout=HEALTH_CHECK_TIMEOUT)
            response.raise_for_status() # Lanza excepción para códigos de status 4xx/5xx
            data = response.json()

            # Verifica si el servicio reporta estar saludable en su respuesta
            # Asumimos que el endpoint /api/state (o similar) devuelve is_healthy
            if data.get("is_healthy", False):
                logger.info(f"{service_name} saludable. Estado reportado: {data}")
                return True, data
            else:
                 # Si responde 200 pero is_healthy es False, loguear y reintentar
                 logger.warning(f"{service_name} accesible pero reporta is_healthy: False. Reintentando...")

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout ({HEALTH_CHECK_TIMEOUT}s) al conectar con {service_name} en {state_endpoint}. Intento {attempts + 1}/{HEALTH_CHECK_RETRIES}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Error de conexión con {service_name} en {state_endpoint}: {e}. Intento {attempts + 1}/{HEALTH_CHECK_RETRIES}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error HTTP general con {service_name} en {state_endpoint}: {e}. Intento {attempts + 1}/{HEALTH_CHECK_RETRIES}")
        except json.JSONDecodeError:
             logger.warning(f"Respuesta de {service_name} no es JSON válido. URL: {state_endpoint}, Respuesta: {response.text}. Intento {attempts + 1}/{HEALTH_CHECK_RETRIES}")
        except Exception as e:
             # Captura cualquier otra excepción inesperada
             logger.exception(f"Excepción inesperada verificando {service_name} en {state_endpoint}. Intento {attempts + 1}/{HEALTH_CHECK_RETRIES}")


        attempts += 1
        if attempts < HEALTH_CHECK_RETRIES:
            time.sleep(RETRY_DELAY) # Esperar antes de reintentar

    logger.error(f"FALLO: {service_name} no respondió saludablemente después de {HEALTH_CHECK_RETRIES} reintentos.")
    # Retornar False y los últimos datos si hubo alguna respuesta (aunque no saludable), o un diccionario vacío
    # En caso de error de conexión persistente, data podría no estar definida, manejar eso.
    return False, data if 'data' in locals() else {"error": "No data received"}

# --- Tests de Conectividad ---

# Opcional: Fixture para una espera inicial antes de todos los tests
# @pytest.fixture(scope="module", autouse=True)
# def initial_wait_for_services():
#     """Espera un tiempo al inicio del módulo para dar tiempo a los servicios a arrancar."""
#     wait_time = int(os.environ.get("SERVICE_STARTUP_WAIT", 15)) # Espera configurable
#     logger.info(f"Esperando {wait_time} segundos para que los servicios se inicien completamente...")
#     time.sleep(wait_time)
#     logger.info("Espera inicial finalizada. Iniciando health checks.")

def test_ecu_is_accessible_and_healthy():
    """
    Verificar que el servicio Matriz ECU es accesible, reporta salud,
    y su estado inicial (la matriz) tiene la estructura esperada.
    """
    # La función check_service_health ahora retorna también los datos de la respuesta
    is_healthy, data = check_service_health("Matriz ECU", ECU_URL)

    # Aserción básica de salud
    assert is_healthy, f"El servicio Matriz ECU no está saludable o accesible después de reintentos. Última respuesta: {data}"

    # --- Aserciones Detalladas sobre el estado de la Matriz ---
    # Asumimos que 'data' contiene el JSON retornado por /api/ecu/state
    logger.info("Verificando estructura del estado de Matriz ECU...")

    # 1. Verificar que la clave 'data' existe en la respuesta principal
    assert "data" in data, f"La respuesta de {ECU_URL}/api/ecu/state no contiene la clave 'data'. Respuesta completa: {data}"

    # Extraer el estado de la matriz
    ecu_state_data = data["data"]

    # 2. Verificar que la clave 'matrix_state' existe dentro de 'data'
    assert "matrix_state" in ecu_state_data, f"La clave 'matrix_state' no se encuentra en los datos del estado de Matriz ECU. Datos: {ecu_state_data}"

    # Obtener la matriz como lista anidada
    matrix_list = ecu_state_data["matrix_state"]

    # 3. Verificar que matrix_state es una lista (representando las capas)
    assert isinstance(matrix_list, list), f"matrix_state no es una lista. Tipo: {type(matrix_list)}. Datos: {ecu_state_data}"

    # 4. Verificar las dimensiones (esto requiere conocer las dimensiones esperadas)
    # Estas dimensiones deberían coincidir con NUM_CAPAS, NUM_FILAS, NUM_COLUMNAS en matriz_ecu.py
    # Puedes obtenerlas del código o de variables de entorno si las usas
    # Por defecto en matriz_ecu.py: NUM_CAPAS=3, NUM_FILAS=10, NUM_COLUMNAS=20
    expected_capas = 3
    expected_filas = 10
    expected_columnas = 20
    expected_vector_dim = 2 # [vx, vy]

    assert len(matrix_list) == expected_capas, f"Número incorrecto de capas en matrix_state. Esperado: {expected_capas}, Obtenido: {len(matrix_list)}. Datos: {ecu_state_data}"

    if expected_capas > 0:
        # Verificar las dimensiones de las filas dentro de cada capa
        # Asumimos que todas las capas tienen el mismo número de filas
        assert isinstance(matrix_list[0], list), f"Las capas no contienen listas (filas). Tipo del primer elemento: {type(matrix_list[0])}. Datos: {ecu_state_data}"
        assert len(matrix_list[0]) == expected_filas, f"Número incorrecto de filas en matrix_state. Esperado: {expected_filas}, Obtenido: {len(matrix_list[0])}. Datos: {ecu_state_data}"

        if expected_filas > 0:
            # Verificar las dimensiones de las columnas dentro de cada fila
            # Asumimos que todas las filas tienen el mismo número de columnas
            assert isinstance(matrix_list[0][0], list), f"Las filas no contienen listas (columnas). Tipo del primer elemento: {type(matrix_list[0][0])}. Datos: {ecu_state_data}"
            assert len(matrix_list[0][0]) == expected_columnas, f"Número incorrecto de columnas en matrix_state. Esperado: {expected_columnas}, Obtenido: {len(matrix_list[0][0])}. Datos: {ecu_state_data}"

            if expected_columnas > 0:
                # Verificar las dimensiones de los vectores [vx, vy]
                # Asumimos que todas las columnas contienen un vector de 2 elementos
                assert isinstance(matrix_list[0][0][0], list), f"Las columnas no contienen listas (vectores). Tipo del primer elemento: {type(matrix_list[0][0][0])}. Datos: {ecu_state_data}"
                assert len(matrix_list[0][0][0]) == expected_vector_dim, f"Dimensión incorrecta del vector [vx, vy]. Esperado: {expected_vector_dim}, Obtenido: {len(matrix_list[0][0][0])}. Datos: {ecu_state_data}"
                # Opcional: Verificar que los elementos del vector son números (int o float)
                assert all(isinstance(val, (int, float)) for val in matrix_list[0][0][0]), f"Los elementos del vector [vx, vy] no son números. Primer vector: {matrix_list[0][0][0]}. Datos: {ecu_state_data}"

    # 5. Opcional: Verificar un valor inicial esperado (ej. ceros si se inicializa así)
    # Aquí verificamos que el primer elemento del vector en [0,0,0] es 0.0
    # Esto puede ser frágil si la simulación se ejecuta muy rápido al inicio
    # Una verificación más robusta sería: verificar que TODOS los valores son 0.0
    # import numpy as np # Asegúrate de importar numpy
    # flat_list = [item for sublist in [item2 for sublist2 in matrix_list for item2 in sublist2] for item in sublist]
    # assert all(abs(x) < 1e-9 for x in flat_list), "No todos los valores iniciales de la matriz son cero."
    # O una verificación más simple del primer elemento:
    if expected_capas > 0 and expected_filas > 0 and expected_columnas > 0:
        assert abs(matrix_list[0][0][0][0]) < 1e-9, f"El primer componente del vector en [0,0,0] no es cero (o cercano). Valor: {matrix_list[0][0][0][0]}. Datos: {ecu_state_data}"
        assert abs(matrix_list[0][0][0][1]) < 1e-9, f"El segundo componente del vector en [0,0,0] no es cero (o cercano). Valor: {matrix_list[0][0][0][1]}. Datos: {ecu_state_data}"

    logger.info("Verificación de estructura y estado inicial de Matriz ECU PASADA.")

def test_malla_watcher_is_accessible_and_healthy():
    """Verificar que el servicio Malla Watcher es accesible y reporta salud."""
    is_healthy, data = check_service_health("Malla Watcher", MALLA_URL)
    assert is_healthy, f"El servicio Malla Watcher no está saludable o accesible. Última respuesta: {data}"
    # Opcional: Aserciones específicas sobre el estado de Malla
    # assert data.get("details", {}).get("mesh", {}).get("initialized") is True, "La malla no está inicializada"
    # assert data.get("details", {}).get("resonator_simulation", {}).get("running") is True, "La simulación de Malla no está corriendo"
    # assert data.get("details", {}).get("mesh", {}).get("num_cells", 0) > 0, "La malla está vacía"

def test_harmony_controller_is_accessible_and_healthy():
    """Verificar que el servicio Harmony Controller es accesible y reporta salud."""
    is_healthy, data = check_service_health("Harmony Controller", HARMONY_CONTROLLER_URL)
    assert is_healthy, f"El servicio Harmony Controller no está saludable o accesible. Última respuesta: {data}"
    # Opcional: Aserciones específicas sobre HC
    # assert data.get("control_loop_running") is True, "El bucle de control de HC no está corriendo"

def test_agent_ai_is_accessible_and_healthy():
    """Verificar que el servicio Agent AI es accesible y reporta salud."""
    is_healthy, data = check_service_health("Agent AI", AGENT_AI_URL)
    assert is_healthy, f"El servicio Agent AI no está saludable o accesible. Última respuesta: {data}"
    # Opcional: Aserciones específicas sobre AgentAI
    # assert data.get("status") == "success", "Agent AI no reporta estado success"

# --- END OF FILE tests/integration/test_integration_connectivity.py ---