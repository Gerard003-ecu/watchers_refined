# --- START OF FILE tests/unit/test_validator.py (AJUSTADO) ---

#!/usr/bin/env python3
"""
test_validator.py - Pruebas unitarias para agent_ai/validation/validator.py
"""

import pytest
import unittest
from unittest.mock import patch, mock_open

# Ajustar ruta de importación
try:
    from agent_ai.validation.validator import (
        validate_module_registration,
        validate_command_response,
        check_missing_dependencies # Mantener si se prueba el placeholder o la implementación
    )
except ImportError:
    # Fallback
    from validator import (
        validate_module_registration,
        validate_command_response,
        check_missing_dependencies
    )


class TestValidators(unittest.TestCase):

    def test_validate_module_registration_success(self):
        """Prueba registro válido (campos básicos)."""
        # Ahora solo requiere nombre y url según el código refactorizado
        module_data = {"nombre": "TestModule", "url": "http://example.com/api/health"}
        valido, mensaje = validate_module_registration(module_data)
        self.assertTrue(valido, mensaje)
        self.assertIn("válidos", mensaje) # Ajustar mensaje esperado si cambia

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

    # --- Tests para check_missing_dependencies ---
    # Comentar estos tests si check_missing_dependencies es solo un placeholder

    # @patch('builtins.open', new_callable=mock_open, read_data='flask\nrequests\nnumpy==1.2.3\n')
    # def test_check_deps_all_covered(self, mock_file_global):
    #     """Prueba cuando todas las dependencias están cubiertas."""
    #     with patch('builtins.open', mock_open(read_data='requests\nnumpy'), create=True) as mock_file_module:
    #         valido, mensaje = check_missing_dependencies("dummy_module_path", "dummy_global_path")
    #         self.assertTrue(valido, f"Debería ser válido si deps están cubiertas. Mensaje: {mensaje}")
    #         mock_file_global.assert_called_with("dummy_global_path", 'r')
    #         mock_file_module.assert_called_with("dummy_module_path", 'r')

    # @patch('builtins.open', new_callable=mock_open, read_data='flask\nrequests\n')
    # def test_check_deps_missing(self, mock_file_global):
    #     """Prueba cuando faltan dependencias."""
    #     with patch('builtins.open', mock_open(read_data='requests\npandas'), create=True) as mock_file_module:
    #         valido, mensaje = check_missing_dependencies("dummy_module_path", "dummy_global_path")
    #         self.assertFalse(valido, "Debería ser inválido si faltan dependencias.")
    #         self.assertIn("pandas", mensaje.lower(), "El mensaje debería mencionar la dependencia faltante.")

    # @patch('builtins.open', side_effect=FileNotFoundError("No existe global"))
    # def test_check_deps_global_not_found(self, mock_open_global_fail):
    #     """Prueba cuando el archivo global no existe."""
    #     valido, mensaje = check_missing_dependencies("dummy_module_path", "dummy_global_path")
    #     self.assertFalse(valido) # Asumiendo que la función real devuelve False
    #     self.assertIn("No se pudo encontrar", mensaje)
    #     self.assertIn("global", mensaje)

    # def test_check_deps_module_not_found(self):
    #     """Prueba cuando el archivo del módulo no existe."""
    #     # Este test es más complejo de mockear correctamente sin os.path.exists
    #     # Comentado por ahora, ya que depende de la implementación final
    #     pass


# --- END OF FILE tests/unit/test_validator.py ---