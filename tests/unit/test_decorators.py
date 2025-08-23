import os
import sys
import time
import unittest
from unittest.mock import patch

# Añadir el directorio raíz al path para que encuentre el módulo 'common'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.decorators import measure_performance


class TestMeasurePerformanceDecorator(unittest.TestCase):
    @patch("common.decorators.requests.post")
    def test_decorator_sends_metrics_and_preserves_metadata(self, mock_post):
        """
        Prueba que el decorador envía las métricas correctamente y preserva los metadatos.
        """
        agent_ai_url = "http://fake-agent-ai:9000"

        @measure_performance(agent_ai_url=agent_ai_url, source_service="test_service")
        def dummy_function(x, y):
            """Docstring de prueba."""
            time.sleep(0.01)
            return x + y

        # 1. Verificar que los metadatos de la función se preservan
        self.assertEqual(dummy_function.__name__, "dummy_function")
        self.assertEqual(dummy_function.__doc__, "Docstring de prueba.")

        # 2. Ejecutar la función y verificar el resultado
        result = dummy_function(5, 10)
        self.assertEqual(result, 15)

        # 3. Dar un pequeño margen para que el hilo de envío de métricas se ejecute
        time.sleep(0.1)

        # 4. Verificar que requests.post fue llamado una vez
        mock_post.assert_called_once()

        # 5. Verificar los argumentos de la llamada a post
        args, kwargs = mock_post.call_args
        expected_url = f"{agent_ai_url}/api/metrics"
        self.assertEqual(args[0], expected_url)

        payload = kwargs["json"]
        self.assertEqual(payload["function_name"], "dummy_function")
        self.assertEqual(payload["source_service"], "test_service")
        self.assertAlmostEqual(payload["execution_time"], 0.01, delta=0.015)
        self.assertEqual(payload["call_count"], 1)

    @patch("common.decorators.requests.post")
    def test_call_count_is_correct_across_multiple_calls(self, mock_post):
        """
        Prueba que el contador de llamadas se incrementa correctamente en llamadas sucesivas.
        """
        agent_ai_url = "http://fake-agent-ai:9000"

        @measure_performance(
            agent_ai_url=agent_ai_url, source_service="counting_service"
        )
        def function_for_counting():
            pass

        # Realizar varias llamadas
        function_for_counting()
        time.sleep(0.1)  # Esperar al hilo

        function_for_counting()
        time.sleep(0.1)  # Esperar al hilo

        function_for_counting()
        time.sleep(0.1)  # Esperar al hilo

        # Verificar que post fue llamado 3 veces
        self.assertEqual(mock_post.call_count, 3)

        # Verificar que el 'call_count' en el payload es correcto en cada llamada
        first_call_payload = mock_post.call_args_list[0].kwargs["json"]
        second_call_payload = mock_post.call_args_list[1].kwargs["json"]
        third_call_payload = mock_post.call_args_list[2].kwargs["json"]

        self.assertEqual(first_call_payload["call_count"], 1)
        self.assertEqual(second_call_payload["call_count"], 2)
        self.assertEqual(third_call_payload["call_count"], 3)


if __name__ == "__main__":
    unittest.main()
