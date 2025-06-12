# --- START OF FILE tests/unit/test_logger.py---

import unittest
import unittest.mock as mock
import logging
import os
import io
from pathlib import Path
import importlib
import sys
import tempfile

LOG_DIR_ENV = os.environ.get("WATCHERS_LOG_DIR", "logs")
base_dir = os.getcwd()
LOG_DIR = os.path.join(base_dir, LOG_DIR_ENV)

# Importar el módulo logger desde su ubicación correcta
try:
    from agent_ai.utils import logger as agent_logger
except ImportError:
    try:
        import logger as agent_logger
    except ImportError:
        raise ImportError(
            "No se pudo importar el módulo logger" 
            "Verifica PYTHONPATH y estructura."
        )

logger_instance = logging.getLogger("agent_ai")


class TestLogger(unittest.TestCase):

    def setUp(self):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self.temp_dir_obj.name)
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

        # Limpiar handlers ANTES de recargar
        for handler in logger_instance.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger_instance.removeHandler(handler)
        logger_instance.handlers.clear()
        logger_instance.propagate = True  # Resetear propagate

        # Recargar el módulo
        importlib.reload(agent_logger)
        self.temp_logger_module = agent_logger
        self.log_file_path = Path(self.temp_logger_module.LOG_FILE)

    def tearDown(self):
        # Limpiar handlers DESPUÉS de tests
        for handler in logger_instance.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
            logger_instance.removeHandler(handler)
        logger_instance.handlers.clear()
        os.chdir(self.original_cwd)
        self.temp_dir_obj.cleanup()

    def test_logger_configuration(self):
        """
        Verifica la configuración
        básica del logger
        """
        log = self.temp_logger_module.get_logger()
        self.assertEqual(log.name, "agent_ai")
        self.assertEqual(log.level, logging.DEBUG)

    def test_file_handler_exists(self):
        """
        Comprueba que el handler de archivo
        esté correctamente configurado
        """
        log = self.temp_logger_module.get_logger()
        file_handlers = [
            h for h in log.handlers 
            if isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(
            len(file_handlers), 1,
            f"Se encontraron {len(file_handlers)} file handlers, se esperaba 1."
        )
        file_handler = file_handlers[0]
        self.assertEqual(file_handler.level, logging.DEBUG)

        # Comparar con el subdirectorio 'logs'
        expected_log_dir = self.temp_dir / self.temp_logger_module.LOG_DIR_ENV
        self.assertEqual(
            self.log_file_path.parent.resolve(), 
            expected_log_dir.resolve(),
            f"El directorio del log ({self.log_file_path.parent}) "
            f"no coincide con el esperado ({expected_log_dir})"
        )

        self.assertTrue(str(self.log_file_path).endswith("agent_ai.log"))
        self.assertEqual(file_handler.mode, "a")

    def test_console_handler_exists(self):
        """
        Verifica el handler de consola con nivel INFO
        """
        log = self.temp_logger_module.get_logger()
        console_handlers = [
            h for h in log.handlers 
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        self.assertEqual(
            len(console_handlers), 1,
            f"Se encontraron {len(console_handlers)} console handlers, "
            "se esperaba 1."
        )
        console_handler = console_handlers[0]
        self.assertEqual(console_handler.level, logging.INFO)
        self.assertIsInstance(console_handler.formatter, logging.Formatter)
        # Verificar que usa stderr
        self.assertIs(console_handler.stream, sys.stderr)

    def test_log_file_creation_and_content(self):
        """
        Verifica que los mensajes
        se escriban en el archivo de log
        """
        log = self.temp_logger_module.get_logger()
        test_message_info = "Mensaje INFO para archivo"
        test_message_debug = "Mensaje DEBUG para archivo"
        log.info(test_message_info)
        log.debug(test_message_debug)

        # Asegurarse de que los handlers hayan tenido tiempo de escribir
        for handler in log.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()

        self.assertTrue(
            self.log_file_path.exists(), 
            "El archivo de log no fue creado"
        )
        content = self.log_file_path.read_text()
        self.assertIn(test_message_info, content)
        self.assertIn(test_message_debug, content)
        self.assertIn("[INFO]", content)
        self.assertIn("[DEBUG]", content)

    @mock.patch('sys.stderr', new_callable=io.StringIO)
    def test_console_output(self, mock_stderr):
        """
        Verifica que los mensajes de nivel 
        INFO y superior se muestren en stderr
        """
        log = self.temp_logger_module.get_logger()
        # Encontrar y remover el handler de consola original
        original_console_handlers = [
            h for h in log.handlers 
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        for h in original_console_handlers:
            log.removeHandler(h)

        # Crear y añadir un NUEVO handler de consola que use el mock_stderr
        formatter = (
            original_console_handlers[0].formatter 
            if original_console_handlers 
            else logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        )
        mock_console_handler = logging.StreamHandler(mock_stderr)
        mock_console_handler.setLevel(logging.INFO)
        mock_console_handler.setFormatter(formatter)
        log.addHandler(mock_console_handler)

        test_message_info = "Mensaje INFO para consola"
        test_message_warning = "Mensaje WARNING para consola"
        test_message_debug = "Mensaje DEBUG NO consola"

        log.debug(test_message_debug)
        log.info(test_message_info)
        log.warning(test_message_warning)

        output = mock_stderr.getvalue()
        self.assertIn(test_message_info, output)
        self.assertIn(test_message_warning, output)
        self.assertNotIn(test_message_debug, output)
        self.assertIn("[INFO]", output)
        self.assertIn("[WARNING]", output)
        self.assertNotIn("[DEBUG]", output)

        # Limpieza: remover el handler mockeado
        log.removeHandler(mock_console_handler)

    def test_get_logger_singleton(self):
        """
        Comprueba que get_logger
        devuelve la misma instancia
        """
        log1 = self.temp_logger_module.get_logger()
        log2 = self.temp_logger_module.get_logger()
        self.assertIs(
            log1,
            log2,
            "Se esperaba una única instancia del logger"
        )

    def test_log_directory_creation(self):
        """
        Verifica que el directorio de logs
        se crea automáticamente
        """
        log_dir = Path(self.temp_logger_module.LOG_DIR)
        self.assertTrue(log_dir.exists(), "El directorio de logs no fue creado")
        self.assertTrue(log_dir.is_dir())

# --- END OF FILE tests/unit/test_logger.py ---