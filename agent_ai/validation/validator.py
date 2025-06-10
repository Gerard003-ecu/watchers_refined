#!/usr/bin/env python3
# --- START OF FILE agent_ai/validation/validator.py ---
"""
validator.py

Funciones de validación en tiempo de ejecución para agent_ai.
"""
import logging
# import os  # Necesario si se implementa check_missing_dependencies -> F401

logger = logging.getLogger("agent_ai.validator")


def validate_module_registration(module_data):
    """
    Verifica que el módulo a registrar contenga los campos obligatorios:
      - nombre, url.
      # - requirements_path (Opcional, temporalmente eliminado)

    Retorna:
      (True, "Módulo válido") si la validación es exitosa;
      (False, mensaje) si falta alguno de los campos requeridos.
    """
    # Ajustar campos requeridos según la implementación final
    required_fields = ["nombre", "url"]
    # Ejemplo de campos
    missing = [
        field
        for field in required_fields
        if field not in module_data or not module_data[field]
    ]
    if missing:
        return (
            False,
            f"Faltan campos obligatorios en registro: {', '.join(missing)}",
        )
    # Aquí iría la llamada a check_missing_dependencies si se implementa
    return True, "Datos de registro básicos válidos"


def validate_command_response(response_json):
    """
    Verifica que la respuesta JSON (de un comando estratégico) sea un dict
    y opcionalmente contenga 'status'. (Puede ser más flexible aquí).

    Retorna:
      (True, "Respuesta válida") si la validación es exitosa;
      (False, mensaje) en caso contrario.
    """
    if not isinstance(response_json, dict):
        return False, "Respuesta no es un diccionario JSON válido"
    # Podríamos no requerir 'status' para todos los comandos estratégicos
    # if "status" not in response_json:
    #     return False, "Respuesta JSON sin clave 'status'"
    return True, "Respuesta válida"


# --- Placeholder o Implementación Futura ---
def check_missing_dependencies(module_req_path, global_req_path):
    """
    (Placeholder) Compara dependencias de un módulo con las globales.
    Debería leer ambos archivos y verificar si las dependencias del módulo
    están satisfechas por el archivo global.
    """
    logger.warning(
        "Validación de dependencias no implementada (%s vs %s). Asumiendo OK.",
        module_req_path, global_req_path
    )
    # Aquí iría la lógica real de comparación de archivos requirements.txt
    # Ejemplo muy básico (no robusto):
    # try:
    #     with open(global_req_path, 'r') as f_global:
    #         global_deps = {
    #             line.strip().split('==')[0].lower() for line in f_global
    #             if line.strip() and not line.startswith('#')
    #         }
    #     with open(module_req_path, 'r') as f_module:
    #         module_deps = {
    #             line.strip().split('==')[0].lower() for line in f_module
    #             if line.strip() and not line.startswith('#')
    #         }
    #     missing = module_deps - global_deps
    #     if missing:
    #         return False, (
    #             f"Dependencias faltantes en entorno global: "
    #             f"{', '.join(missing)}"
    #         )
    #     return True, "Dependencias cubiertas"
    # except FileNotFoundError as e:
    #     logger.error(f"Error al leer archivo de dependencias: {e}")
    #     return False, f"No se pudo encontrar archivo: {e.filename}"
    # except Exception as e:
    #     logger.error(f"Error inesperado al verificar dependencias: {e}")
    #     return False, "Error al verificar dependencias"
    return True, "Validación de dependencias omitida"


# --- END OF FILE agent_ai/validation/validator.py ---
