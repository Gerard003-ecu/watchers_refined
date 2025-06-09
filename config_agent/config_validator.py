#!/usr/bin/env python3
"""
config_validator.py

Funciones para validar archivos de configuración del ecosistema watchers:
  - Validar que un Dockerfile cumpla con ciertos criterios básicos.
  - Validar la estructura del archivo docker-compose.yml.
  - Validar que el archivo de requirements no esté vacío y cumpla con el
    formato esperado.

Estas funciones ayudan al config_agent a asegurar que la infraestructura
esté correctamente configurada.
"""

import yaml  # Necesitas PyYAML
from common.validators_common import file_exists, read_file_lines


def validate_dockerfile(dockerfile_path: str):
    """
    Valida que el Dockerfile contenga instrucciones básicas (FROM y CMD).
    Retorna (True, mensaje) si es válido, de lo contrario (False, mensaje).
    """
    if not file_exists(dockerfile_path):
        return False, "Dockerfile no encontrado."

    lines = read_file_lines(dockerfile_path)
    if not any(line.strip().startswith("FROM") for line in lines):
        return False, "Falta instrucción FROM."
    if not any(line.strip().startswith("CMD") for line in lines):
        return False, "Falta instrucción CMD."
    return True, "Dockerfile válido."


def validate_docker_compose(compose_path: str):
    """
    Valida la estructura del archivo docker-compose.yml.
    Verifica que tenga una versión y una sección de servicios.
    """
    if not file_exists(compose_path):
        return False, "docker-compose.yml no encontrado."
    try:
        with open(compose_path, "r") as f:
            compose_data = yaml.safe_load(f)
        if "version" not in compose_data:
            return False, "Falta la clave 'version'."
        if "services" not in compose_data:
            return False, "Falta la sección 'services'."
        return True, "docker-compose.yml válido."
    except Exception as e:
        return False, f"Error parseando docker-compose.yml: {e}"


def validate_requirements_file(req_path: str):
    """
    Valida que el archivo de requirements (generado globalmente) exista y no
    esté vacío.
    """
    if not file_exists(req_path):
        return False, "Archivo requirements.txt no encontrado."
    lines = read_file_lines(req_path)
    if not lines:
        return False, "Archivo requirements.txt está vacío."
    return True, "Archivo requirements.txt válido."
