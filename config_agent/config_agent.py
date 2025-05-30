#!/usr/bin/env python3
"""
config_agent.py

Este módulo orquesta la validación de la configuración del ecosistema watchers.
Utiliza funciones del módulo config_validator para:
  - Validar Dockerfiles.
  - Validar el archivo docker-compose.yml.
  - Validar archivos de requirements.
  
El objetivo es asegurar que la infraestructura esté correctamente configurada
antes de desplegar el ecosistema.

Además, este módulo envía la "señal" de configuración (los resultados de validación)
a agent_ai mediante un endpoint REST, de modo que se pueda tomar acción si es necesario.
"""

import os
import logging
import json
import requests

from config_agent.config_validator import (
    validate_dockerfile,
    validate_docker_compose,
    validate_requirements_file
)

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("config_agent")

# Endpoint de agent_ai para recibir actualizaciones de configuración
AGENT_AI_CONFIG_ENDPOINT = os.getenv("AGENT_AI_CONFIG_ENDPOINT", "http://agent_ai:9000/api/config")

def validate_configurations():
    """
    Valida los principales archivos de configuración:
      - Dockerfile principal (en la raíz o de cada módulo).
      - docker-compose.yml.
      - Archivo global de requirements (requirements.txt).
    Retorna un diccionario con el resultado de cada validación.
    """
    results = {}

    # Validar Dockerfile principal
    dockerfile_path = os.path.join(os.getcwd(), "Dockerfile")
    results["Dockerfile"] = validate_dockerfile(dockerfile_path)

    # Validar docker-compose.yml
    compose_path = os.path.join(os.getcwd(), "docker-compose.yml")
    results["docker-compose.yml"] = validate_docker_compose(compose_path)

    # Validar archivo global de requirements
    requirements_path = os.path.join(os.getcwd(), "requirements.txt")
    results["requirements.txt"] = validate_requirements_file(requirements_path)

    return results

def send_config_status(results: dict) -> bool:
    """
    Envía la señal de configuración a agent_ai mediante un POST a AGENT_AI_CONFIG_ENDPOINT.
    Se envía un JSON con los resultados de validación.
    
    Retorna True si la señal se envió correctamente, o False en caso de error.
    """
    try:
        payload = {"config_status": results}
        response = requests.post(AGENT_AI_CONFIG_ENDPOINT, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Configuración enviada a agent_ai: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"Error enviando la configuración a agent_ai: {e}")
        return False

def main():
    logger.info("Iniciando validación de configuraciones...")
    results = validate_configurations()
    for file, (status, msg) in results.items():
        logger.info(f"Validación de {file}: {status} - {msg}")
    
    if any(status is False for status, _ in results.values()):
        logger.error("Existen problemas de configuración. Se requiere intervención.")
    else:
        logger.info("Todas las configuraciones son correctas.")

    # Enviar la señal de configuración a agent_ai
    if send_config_status(results):
        logger.info("La señal de configuración se envió correctamente.")
    else:
        logger.error("No se pudo enviar la señal de configuración a agent_ai.")

if __name__ == "__main__":
    main()
