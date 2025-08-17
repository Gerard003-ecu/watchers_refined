#!/usr/bin/env python3
"""
config_validator.py - Analizador de Topología y Configuración

Contiene funciones puras para cargar y validar los archivos de configuración
del ecosistema watchers, incluyendo la topología, los Dockerfiles,
las dependencias y la Matriz de Interacción Central (MIC).
"""
import os
import subprocess
import yaml
from typing import Tuple, Dict, Any, List

from common.validators_common import file_exists

def load_yaml_file(file_path: str) -> Tuple[bool, Dict[str, Any] | str]:
    """Carga un archivo YAML de forma segura."""
    if not file_exists(file_path):
        return False, f"Archivo no encontrado: {file_path}"
    try:
        with open(file_path, "r") as f:
            return True, yaml.safe_load(f)
    except Exception as e:
        return False, f"Error al parsear el archivo YAML {file_path}: {e}"

def validate_topology(topology_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Valida la estructura básica del archivo de topología."""
    if "services" not in topology_data:
        return False, "Falta la sección 'services' en la topología."
    if "mic" not in topology_data:
        return False, "Falta la sección 'mic' (Matriz de Interacción Central) en la topología."
    return True, "Estructura de topología válida."

def validate_dockerfile_best_practices(dockerfile_path: str) -> Tuple[bool, str]:
    """Valida un Dockerfile contra mejores prácticas (FROM, CMD, USER, EXPOSE)."""
    if not file_exists(dockerfile_path):
        return False, "Dockerfile no encontrado."

    with open(dockerfile_path, "r") as f:
        lines = [line.strip() for line in f.readlines()]

    checks = {
        "FROM": any(line.startswith("FROM") for line in lines),
        "CMD": any(line.startswith("CMD") for line in lines),
        "USER": any(line.startswith("USER") and "appuser" in line for line in lines),
        "EXPOSE": any(line.startswith("EXPOSE") for line in lines),
    }

    missing = [key for key, found in checks.items() if not found]
    if missing:
        return False, f"Dockerfile incompleto. Faltan las instrucciones: {', '.join(missing)}."

    return True, "Dockerfile sigue las mejores prácticas básicas."

def check_dependency_consistency(req_in_path: str) -> Tuple[bool, str]:
    """
    Verifica si requirements.txt está sincronizado con requirements.in
    usando pip-compile --dry-run.
    """
    req_txt_path = req_in_path.replace(".in", ".txt")
    if not file_exists(req_in_path):
        return False, "requirements.in no encontrado."
    if not file_exists(req_txt_path):
        return False, "requirements.txt no encontrado. Por favor, compile."

    try:
        # Usamos --quiet para una salida más limpia
        result = subprocess.run(
            ["pip-compile", "--dry-run", "--quiet", req_in_path],
            capture_output=True, text=True, check=True
        )
        return True, "requirements.txt está sincronizado."
    except subprocess.CalledProcessError:
        return False, "requirements.txt está desactualizado. Por favor, recompile."
    except FileNotFoundError:
        return False, "'pip-compile' no encontrado. Asegúrese de que pip-tools esté instalado."

def validate_mic(mic_permissions: Dict[str, Any], observed_interactions: Dict[str, List[str]]) -> Tuple[bool, List[str]]:
    """
    Valida las interacciones observadas contra la MIC de permisos.
    """
    violations = []
    all_ok = True
    for source, destinations in observed_interactions.items():
        if source not in mic_permissions:
            violations.append(f"VIOLACIÓN: El servicio '{source}' no tiene permisos definidos en la MIC pero intenta comunicarse.")
            all_ok = False
            continue

        allowed_destinations = mic_permissions[source]
        for dest in destinations:
            if dest not in allowed_destinations:
                violations.append(f"VIOLACIÓN: Interacción no permitida detectada: '{source}' -> '{dest}'.")
                all_ok = False

    if all_ok:
        return True, ["MIC consistente con las interacciones observadas."]
    return False, violations
