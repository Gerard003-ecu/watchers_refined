#!/usr/bin/env python3
"""
config_agent.py - Constructor de Topología del Ecosistema Watchers

Este agente actúa como un "functor" que mapea la configuración física
(docker-compose, Dockerfiles) a un modelo algebraico-topológico
(la Matriz de Interacción Central y la taxonomía de servicios).

Flujo de trabajo:
1. Carga la topología deseada desde config/ecosystem_topology.yml.
2. Carga la configuración de despliegue desde docker-compose.yml.
3. Para cada servicio a desplegar:
    a. Valida sus artefactos (Dockerfile, dependencias).
    b. Lo clasifica según la topología.
    c. Descubre sus dependencias de red (interacciones).
4. Valida las interacciones observadas contra la MIC.
5. Envía un informe completo y estructurado a agent_ai.
"""
import os
import logging
import requests
import re
import time
from typing import Dict, Any, List

from config_agent.config_validator import (
    load_yaml_file,
    validate_topology,
    validate_dockerfile_best_practices,
    check_dependency_consistency,
    validate_mic,
)

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("config_agent")

AGENT_AI_CONFIG_ENDPOINT = os.getenv("AGENT_AI_CONFIG_ENDPOINT", "http://agent_ai:9000/api/config_report")
TOPOLOGY_PATH = "config/ecosystem_topology.yml"
COMPOSE_PATH = "docker-compose.yml"

def discover_interactions(service_data: Dict[str, Any]) -> List[str]:
    """Parsea las variables de entorno de un servicio para encontrar URLs de otros servicios."""
    interactions = []
    env_vars = service_data.get("environment", [])
    if not env_vars:
        return interactions

    # Busca patrones como scheme://user:pass@host:port/path
    url_pattern = re.compile(r"\w+://(?:[^@]+@)?([^:/]+)")
    for var in env_vars:
        match = url_pattern.search(var)
        if match:
            interactions.append(match.group(1))
    return list(set(interactions)) # Devuelve destinos únicos

def build_report():
    """Construye el informe de configuración completo."""
    report = {"global_status": "OK", "services": {}, "mic_validation": {}}

    # 1. Cargar y validar topología y compose
    topology_ok, topology_data = load_yaml_file(TOPOLOGY_PATH)
    compose_ok, compose_data = load_yaml_file(COMPOSE_PATH)

    if not (topology_ok and compose_ok):
        report["global_status"] = "ERROR"
        report["error"] = "No se pudieron cargar los archivos de configuración principales."
        return report

    status, msg = validate_topology(topology_data)
    if not status:
        report["global_status"] = "ERROR"
        report["error"] = f"Topología inválida: {msg}"
        return report

    defined_services = topology_data.get("services", {})
    mic_permissions = topology_data.get("mic", {})
    deployed_services = compose_data.get("services", {})
    observed_interactions = {}

    # 2. Analizar cada servicio a desplegar
    for name, data in deployed_services.items():
        if name not in defined_services:
            logger.warning(f"El servicio '{name}' se está desplegando pero no está definido en la topología.")
            continue

        context = data.get("build", {}).get("context", ".")
        dockerfile_path = os.path.join(context, data.get("build", {}).get("dockerfile", "Dockerfile"))
        req_in_path = os.path.join(context, "requirements.in")

        service_report = {
            "type": defined_services[name].get("type"),
            "category": defined_services[name].get("category"),
            "dockerfile_status": validate_dockerfile_best_practices(dockerfile_path),
            "dependency_status": check_dependency_consistency(req_in_path),
            "observed_interactions": discover_interactions(data),
        }
        report["services"][name] = service_report
        observed_interactions[name] = service_report["observed_interactions"]

    # 3. Validar la MIC
    mic_ok, mic_messages = validate_mic(mic_permissions, observed_interactions)
    report["mic_validation"] = {"status": "OK" if mic_ok else "VIOLATION", "messages": mic_messages}

    # 4. Determinar el estado global
    for service in report["services"].values():
        if not (service["dockerfile_status"][0] and service["dependency_status"][0]):
            report["global_status"] = "ERROR"
            break
    if not mic_ok:
        report["global_status"] = "ERROR"

    return report

def send_report(report: Dict[str, Any], retries: int = 3, delay: int = 5):
    """Envía el informe a agent_ai, con reintentos en caso de fallo."""
    for attempt in range(retries):
        try:
            response = requests.post(AGENT_AI_CONFIG_ENDPOINT, json=report, timeout=15)
            response.raise_for_status()
            logger.info(f"Informe de configuración enviado a agent_ai. Respuesta: {response.status_code}")
            return  # Envío exitoso, salimos de la función
        except requests.exceptions.RequestException as e:
            logger.warning(f"Intento {attempt + 1}/{retries} fallido al enviar el informe: {e}")
            if attempt < retries - 1:
                logger.info(f"Reintentando en {delay} segundos...")
                time.sleep(delay)
            else:
                logger.error(f"No se pudo enviar el informe de configuración a agent_ai después de {retries} intentos.")
                break # Salir del bucle después del último intento

def main():
    logger.info("Iniciando constructor de topología y validación...")
    report = build_report()

    # Imprimir un resumen legible
    logger.info("--- Resumen de Validación ---")
    logger.info(f"Estado Global: {report['global_status']}")
    for name, data in report["services"].items():
        logger.info(f"  - Servicio: {name}")
        logger.info(f"    - Dockerfile: {'OK' if data['dockerfile_status'][0] else 'FAIL'}")
        logger.info(f"    - Dependencias: {'OK' if data['dependency_status'][0] else 'FAIL'}")
    logger.info(f"Validación de MIC: {report['mic_validation']['status']}")
    for msg in report['mic_validation']['messages']:
        logger.info(f"  - {msg}")

    send_report(report)
    logger.info("Proceso de config_agent finalizado.")

if __name__ == "__main__":
    main()
