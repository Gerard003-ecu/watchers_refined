# --- START OF FILE tests/unit/test_validator.py (AJUSTADO) ---

# !/usr/bin/env python3
"""test_validator.py - Pruebas unitarias para validation/validator.py."""

import unittest

# Ajustar ruta de importación
try:
    from agent_ai.validation.validator import (
        validate_command_response,
        validate_module_registration,
    )
except ImportError:
    # Fallback
    from validator import validate_command_response, validate_module_registration


class TestValidators(unittest.TestCase):
    def test_validate_module_registration_success(self):
        """Prueba registro válido (campos básicos)."""
        # Solo requiere nombre y url según el código refactorizado
        module_data = {"nombre": "TestModule", "url": "http://example.com/api/health"}
        valido, mensaje = validate_module_registration(module_data)
        self.assertTrue(valido, mensaje)
        # Ajustar mensaje esperado si cambia
        self.assertIn("válidos", mensaje)

    def test_validate_module_registration_missing_name(self):
        """Prueba fallo por nombre faltante."""
        module_data = {"url": "http://example.com/api/health"}
        valido, mensaje = validate_module_registration(module_data)
        self.assertFalse(valido)
        self.assertIn("nombre", mensaje)

    def test_validate_module_registration_missing_url(self):
        """Prueba fallo por URL faltante."""
        module_data = {"nombre": "TestModule"}
        valido, mensaje = validate_module_registration(module_data)
        self.assertFalse(valido)
        self.assertIn("url", mensaje)

    def test_validate_module_registration_empty_url(self):
        """Prueba fallo por URL vacía."""
        module_data = {"nombre": "TestModule", "url": ""}
        valido, mensaje = validate_module_registration(module_data)
        self.assertFalse(valido)
        self.assertIn("url", mensaje)

    def test_validate_command_response_success(self):
        """Prueba respuesta de comando válida (dict)."""
        response_data = {"status": "success", "data": {}}
        valido, mensaje = validate_command_response(response_data)
        self.assertTrue(valido)
        self.assertIn("válida", mensaje)

    def test_validate_command_response_not_dict(self):
        """Prueba fallo si la respuesta no es un diccionario."""
        response_data = "un string"
        valido, mensaje = validate_command_response(response_data)
        self.assertFalse(valido)
        self.assertIn("diccionario", mensaje)

    def test_validate_command_response_list(self):
        """Prueba fallo si la respuesta es una lista."""
        response_data = ["a", "b"]
        valido, mensaje = validate_command_response(response_data)
        self.assertFalse(valido)
        self.assertIn("diccionario", mensaje)

    # Tests para check_missing_dependencies comentados por ahora
    # Se mantienen como referencia pero no se ejecutan


# --- END OF FILE tests/unit/test_validator.py ---
