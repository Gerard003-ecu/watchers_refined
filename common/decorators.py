import time
import threading
import requests
import functools
import os
import logging

logger = logging.getLogger(__name__)

def measure_performance(agent_ai_url: str, source_service: str):
    """
    Decorador de fábrica que crea un decorador para medir el rendimiento de una función.

    Args:
        agent_ai_url (str): La URL base del servicio AgentAI para enviar las métricas.
        source_service (str): El nombre del servicio de origen de la métrica.
    """
    def decorator(func):
        """
        El decorador real que envuelve la función.
        """
        # Usar un atributo de la función para mantener el contador de llamadas
        # se reiniciará si el proceso se reinicia.
        if not hasattr(decorator, 'call_count'):
            decorator.call_count = 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            El wrapper que mide el tiempo, cuenta las llamadas y envía los datos.
            """
            start_time = time.monotonic()

            # Ejecutar la función original
            result = func(*args, **kwargs)

            end_time = time.monotonic()
            execution_time = end_time - start_time

            decorator.call_count += 1

            # Preparar los datos para enviar
            metrics_payload = {
                "source_service": source_service,
                "function_name": func.__name__,
                "execution_time": execution_time,
                "call_count": decorator.call_count,
            }

            # Enviar los datos en un hilo separado para no bloquear
            metrics_url = f"{agent_ai_url}/api/metrics"
            thread = threading.Thread(
                target=send_metrics,
                args=(metrics_url, metrics_payload),
                daemon=True,
                name=f"MetricsThread-{func.__name__}"
            )
            thread.start()

            return result
        return wrapper
    return decorator

def send_metrics(url: str, payload: dict):
    """
    Función para enviar métricas a AgentAI a través de una solicitud POST.

    Args:
        url (str): La URL completa del endpoint de métricas.
        payload (dict): El diccionario de métricas a enviar como JSON.
    """
    try:
        response = requests.post(url, json=payload, timeout=5.0)
        response.raise_for_status()
        logger.debug(f"Métricas enviadas exitosamente a {url}. Payload: {payload}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al enviar métricas a {url}: {e}")
