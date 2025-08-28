import logging
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class ApiClient:
    """
    Un cliente de API robusto para la comunicación entre microservicios.

    Esta clase encapsula la lógica de realizar peticiones HTTP (GET, POST)
    con un mecanismo de reintentos incorporado, incluyendo backoff exponencial.
    Está diseñada para ser reutilizada por cualquier servicio que necesite
    comunicarse con otro de manera fiable.

    Attributes:
        max_retries (int): El número máximo de veces que se reintentará una petición.
        base_retry_delay (float): El retardo base en segundos para el primer reintento.
        timeout (float): El tiempo de espera en segundos para las peticiones.
    """

    def __init__(self, max_retries: int, base_retry_delay: float, timeout: float):
        """
        Inicializa el ApiClient.

        Args:
            max_retries (int): Número máximo de reintentos.
            base_retry_delay (float): Retardo base para el backoff exponencial.
            timeout (float): Timeout para las peticiones de requests.
        """
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.timeout = timeout

    def get(
        self, url: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Realiza una petición GET con reintentos.

        Args:
            url (str): La URL a la que se hará la petición.
            params (Optional[Dict[str, Any]], optional): Parámetros de query.

        Returns:
            Optional[Dict[str, Any]]: El JSON de la respuesta si es exitosa,
                                      o None si falla tras todos los reintentos.
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.error(
                    "Error en GET a %s (intento %d/%d): %s",
                    url,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    delay = self.base_retry_delay * (2**attempt)
                    time.sleep(delay)
        return None

    def post(
        self, url: str, json_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Realiza una petición POST con reintentos.

        Args:
            url (str): La URL a la que se hará la petición.
            json_data (Optional[Dict[str, Any]], optional): El payload JSON.

        Returns:
            Optional[Dict[str, Any]]: El JSON de la respuesta si es exitosa,
                                      o None si falla tras todos los reintentos.
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=json_data, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                logger.error(
                    "Error en POST a %s (intento %d/%d): %s",
                    url,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries - 1:
                    delay = self.base_retry_delay * (2**attempt)
                    time.sleep(delay)
        return None
