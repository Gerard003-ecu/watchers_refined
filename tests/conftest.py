# mi-proyecto/tests/conftest.py
import pytest
import requests
import os
import logging

logger = logging.getLogger("pytest_precondition_checks")

# Leer URLs desde variables de entorno, proporcionar valores por defecto
# Estos deben ser los mismos que usas en tus tests de integración
ECU_URL_HEALTH = os.environ.get("TEST_ECU_URL", "http://localhost:8000")
MALLA_URL_HEALTH = os.environ.get("TEST_MALLA_URL", "http://localhost:5001")
# Añade más servicios si es necesario

SERVICES_TO_CHECK = {
    "MatrizECU": ECU_URL_HEALTH,
    "MallaWatcher": MALLA_URL_HEALTH,
}


@pytest.fixture(scope="session", autouse=True)
def check_all_services_health(request):
    """
    Fixture de sesión que verifica la salud de todos los servicios requeridos
    antes de que comience la ejecución de los tests de integración.
    Falla la sesión de test si algún servicio no está saludable.
    Se ejecuta solo para tests dentro del directorio 'integration'.
    """
    # Determinar si los tests que se están ejecutando son de integración
    # Esto se puede hacer inspeccionando los items de la sesión o las rutas.
    # Una forma simple es verificar si alguna ruta de test incluye "integration".
    is_integration_run = False
    for item in request.session.items:
        if "integration" in item.nodeid.lower(): # O item.fspath si prefieres rutas de archivo
            is_integration_run = True
            break

    if not is_integration_run:
        logger.info("No es una ejecución de tests de integración. Saltando verificaciones de salud de servicios.")
        return

    # Permitir saltar esta verificación si se pasa una opción a pytest
    if request.config.getoption("--skip-health-checks", default=False):
        logger.warning(
            "Saltando verificaciones de salud de servicios pre-test para tests de integración."
        )
        return

    logger.info(
        "Iniciando verificación de salud de servicios pre-test para tests de integración..."
    )
    all_healthy = True
    for service_name, base_url in SERVICES_TO_CHECK.items():
        health_url = f"{base_url}/api/health"
        try:
            # Timeout corto para health check
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()  # Lanza excepción para 4xx/5xx

            # Opcional: Verificar el contenido de la respuesta de salud
            # health_data = response.json()
            # if health_data.get("status") not in ["success", "ok", "warning"]:
            #     logger.error(
            #         f"Servicio {service_name} en {health_url}"
            #         f"reportó no saludable: {health_data.get('status')}"
            #     )
            #     all_healthy = False
            # else:
            #     logger.info(
            #         f"Servicio {service_name} en {health_url} "
            #         f"está saludable (status: {health_data.get('status')})."
            #     )

            logger.info(
                f"Servicio {service_name} en {health_url} "
                f"respondió con status {response.status_code}."
            )

        except requests.exceptions.RequestException as e:
            logger.error(
                "FALLO DE PRECONDICIÓN:"
                "No se pudo conectar o verificar la salud"
                f"del servicio {service_name} en {health_url}. Error: {e}"
            )
            all_healthy = False
        except Exception as e:
            logger.error(
                "FALLO DE PRECONDICIÓN:"
                "Error inesperado verificando la salud"
                f"del servicio {service_name} en {health_url}. Error: {e}"
            )
            all_healthy = False

    if not all_healthy:
        pytest.exit(
            "Uno o más servicios requeridos"
            "no están saludables o no son accesibles."
            "Deteniendo tests de integración.",
            returncode=1,
        )
    else:
        logger.info(
            "Todos los servicios requeridos están saludables. "
            "Procediendo con los tests."
        )


def pytest_addoption(parser):
    """
    Permite añadir opciones
    de línea de comando a pytest.
    """
    parser.addoption(
        "--skip-health-checks",
        action="store_true",
        help="Saltar las verificaciones de salud de servicios"
    )
