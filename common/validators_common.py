#!/usr/bin/env python3
"""
validators_common.py

Funciones auxiliares comunes para la validación:
  - file_exists: Verifica si un archivo existe.
  - read_file_lines: Lee todas las líneas de un archivo.
"""


def file_exists(file_path: str) -> bool:
    """
    Retorna True si el archivo en file_path existe, False en caso contrario.
    """
    try:
        with open(file_path, "r"):
            return True
    except FileNotFoundError:
        return False


def read_file_lines(file_path: str) -> list:
    """Retorna una lista de líneas del archivo."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.readlines()
