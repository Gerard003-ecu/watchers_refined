#!/usr/bin/env python3
"""
Punto central de coordinación para el ecosistema de Watchers.

Este módulo define la clase `AgentAI`, responsable de la orquestación
de alto nivel del sistema. Sus responsabilidades incluyen:

- Registrar y monitorizar la salud de los módulos componentes
  (denominados "watchers tools").
- Capturar y procesar la afinidad ('aporta_a') y naturaleza
  ('naturaleza_auxiliar') de estos módulos.
- Notificar al `HarmonyController` sobre los módulos auxiliares
  que se encuentren saludables, junto con su afinidad y naturaleza.
- Monitorizar el estado general del sistema a través del `HarmonyController`.
- Determinar el estado operativo deseado (conocido como "setpoint de armonía").
- Comunicar este setpoint al `HarmonyController`.
- Procesar entradas y comandos provenientes de fuentes externas,
  como `cogniboard` o `config_agent`, para ajustar la estrategia operativa.

El módulo también configura y expone una API Flask para la interacción externa,
permitiendo el registro de módulos, la consulta del estado y
la recepción de comandos.
"""

# Standard library imports
import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

# Third-party imports
import numpy as np
import requests
from flask import Flask, jsonify, request

# Local application imports
from .utils.logger import get_logger
from .validation.validator import (
    check_missing_dependencies,
    validate_module_registration,
)

logger = get_logger()

# --- Configuración ---
HARMONY_CONTROLLER_URL = os.environ.get(
    "HARMONY_CONTROLLER_URL", "http://harmony_controller:7000"
)
HARMONY_CONTROLLER_REGISTER_URL = os.environ.get(
    "HARMONY_CONTROLLER_REGISTER_URL",
    f"{HARMONY_CONTROLLER_URL}/api/harmony/register_tool"
)
# Nombre de la variable ENV
HARMONY_CONTROLLER_URL_ENV = "HARMONY_CONTROLLER_URL"
HARMONY_CONTROLLER_REGISTER_URL_ENV = (
    "HARMONY_CONTROLLER_REGISTER_URL"
)
AGENT_AI_ECU_URL_ENV = "AGENT_AI_ECU_URL"
AGENT_AI_ECU_FIELD_VECTOR_URL_ENV = "AGENT_AI_ECU_FIELD_VECTOR_URL"
AGENT_AI_MALLA_URL_ENV = "AGENT_AI_MALLA_URL"
DEFAULT_HC_URL = "http://harmony_controller:7000"
DEFAULT_ECU_URL = "http://ecu:8000"
DEFAULT_ECU_FIELD_VECTOR_URL = "http://ecu:8000/api/ecu/field_vector"
DEFAULT_MALLA_URL = "http://malla_watcher:5001"
STRATEGIC_LOOP_INTERVAL = float(
    os.environ.get("AA_INTERVAL", 5.0)
)
REQUESTS_TIMEOUT = float(
    os.environ.get("AA_REQUESTS_TIMEOUT", 4.0)
)
GLOBAL_REQUIREMENTS_PATH = os.environ.get(
    "AA_GLOBAL_REQ_PATH", "/app/requirements.txt"
)
MAX_RETRIES = int(os.environ.get("AA_MAX_RETRIES", 3))
BASE_RETRY_DELAY = float(
    os.environ.get("AA_BASE_RETRY_DELAY", 0.5)
)


class AgentAI:
    """Orquesta los módulos del sistema y gestiona la estrategia global.

    Esta clase es el núcleo central de AgentAI. Se encarga de mantener un
    registro de los módulos activos, monitorizar su estado de salud,
    interactuar con el Harmony Controller para obtener el estado del sistema y
    enviar nuevos setpoints de armonía. También procesa información de
    fuentes externas para adaptar su comportamiento estratégico.

    Attributes:
        modules (Dict[str, Dict]): Un diccionario que almacena la información
            de los módulos registrados, usando el nombre del módulo como clave.
        harmony_state (Dict[str, Any]): El último estado conocido recibido del
            Harmony Controller.
        target_setpoint_vector (List[float]): El vector de setpoint de armonía
            que AgentAI intenta alcanzar.
        current_strategy (str): La estrategia operativa actual (por ejemplo,
            'default', 'estabilidad', 'rendimiento').
        external_inputs (Dict[str, Any]): Almacena señales o datos recibidos
            de sistemas externos como Cogniboard o Config Agent.
        lock (threading.Lock): Un cerrojo para sincronizar el acceso a los
            datos compartidos entre hilos.
        central_urls (Dict[str, str]): URLs de los servicios centrales como
            Harmony Controller, ECU y Malla Watcher.
        hc_register_url (str): URL específica para registrar herramientas
            auxiliares en el Harmony Controller.
    """

    def __init__(self):
        """Inicializa una instancia de AgentAI.

        Configura el estado inicial, incluyendo el vector de setpoint objetivo,
        la estrategia actual, las entradas externas, y las URLs de los
        servicios centrales. También prepara el hilo para el bucle estratégico
        principal.
        """
        self.modules: Dict[str, Dict] = {}
        self.harmony_state: Dict[str, Any] = {}
        try:
            initial_vector_str = os.environ.get(
                "AA_INITIAL_SETPOINT_VECTOR", "[1.0, 0.0]"
            )
            parsed_vector = json.loads(initial_vector_str)
            if isinstance(parsed_vector, list) and all(
                isinstance(x, (int, float)) for x in parsed_vector
            ):
                self.target_setpoint_vector: List[float] = parsed_vector
            else:
                raise ValueError(
                    "El valor parseado no es una lista de números")
        except (json.JSONDecodeError, ValueError) as e:
            log_msg = (
                "AA_INITIAL_SETPOINT_VECTOR ('%s') inválido (%s), usando "
                "default [1.0, 0.0]"
            )
            logger.error(
                log_msg, initial_vector_str, e
            )
            self.target_setpoint_vector = [1.0, 0.0]
        self.current_strategy: str = os.environ.get(
            "AA_INITIAL_STRATEGY", "default"
        )
        self.external_inputs: Dict[str, Any] = {
            "cogniboard_signal": None,
            "config_status": None,
        }
        self.lock = threading.Lock()

        # Atributos para el estado del sistema basado en config_agent
        self.system_report: Dict[str, Any] = {}
        self.mic: Dict[str, Any] = {}
        self.service_map: Dict[str, Any] = {}
        self.is_architecture_validated: bool = False
        self.operational_status: str = "STARTING"  # STARTING, OPERATIONAL, DEGRADED, HALTED

        # Almacenamiento para métricas de rendimiento
        self.performance_metrics: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.metrics_lock = threading.Lock()

        self.central_urls: Dict[str, str] = {}
        hc_url = os.environ.get(HARMONY_CONTROLLER_URL_ENV)
        if hc_url:
            hc_url_val = hc_url
        else:
            hc_url_val = DEFAULT_HC_URL
        self.central_urls["harmony_controller"] = hc_url_val

        ecu_url = os.environ.get(AGENT_AI_ECU_URL_ENV)
        self.central_urls["ecu"] = ecu_url if ecu_url else DEFAULT_ECU_URL

        ecu_field_vector_url = os.environ.get(AGENT_AI_ECU_FIELD_VECTOR_URL_ENV)
        self.central_urls["ecu_field_vector"] = (
            ecu_field_vector_url
            if ecu_field_vector_url
            else DEFAULT_ECU_FIELD_VECTOR_URL
        )

        malla_url = os.environ.get(AGENT_AI_MALLA_URL_ENV)
        self.central_urls["malla_watcher"] = (
            malla_url if malla_url else DEFAULT_MALLA_URL
        )

        logger.info(
            "URLs Centrales: HC='%s', ECU='%s', Malla='%s'",
            self.central_urls["harmony_controller"],
            self.central_urls["ecu"],
            self.central_urls["malla_watcher"],
        )

        hc_reg_url_env = os.environ.get(
            HARMONY_CONTROLLER_REGISTER_URL_ENV
        )
        hc_base_url = self.central_urls["harmony_controller"]
        self.hc_register_url = (
            hc_reg_url_env
            if hc_reg_url_env
            else f"{hc_base_url}/api/harmony/register_tool"
        )
        logger.info("URL registro HC: '%s'", self.hc_register_url)
        logger.info("AgentAI inicializado.")

    def store_metric(self, data: Dict[str, Any]):
        """Almacena una métrica de rendimiento de un servicio.

        Args:
            data (Dict[str, Any]): Un diccionario con los datos de la métrica.
                Se esperan las claves: 'source_service', 'function_name',
                'execution_time', 'call_count'.
        """
        source = data.get("source_service")
        func_name = data.get("function_name")

        if not source or not func_name:
            logger.warning("Métrica recibida sin 'source_service' o 'function_name': %s", data)
            return

        metric_record = {
            "timestamp": time.time(),
            "execution_time": data.get("execution_time"),
            "call_count": data.get("call_count"),
        }

        with self.metrics_lock:
            if source not in self.performance_metrics:
                self.performance_metrics[source] = {}
            if func_name not in self.performance_metrics[source]:
                self.performance_metrics[source][func_name] = []

            self.performance_metrics[source][func_name].append(metric_record)
            logger.debug("Métrica de %s/%s almacenada.", source, func_name)

    def update_system_architecture(self, report: Dict[str, Any]):
        """
        Procesa y almacena el informe de arquitectura del sistema de config_agent.

        Args:
            report (Dict[str, Any]): El informe completo de config_agent.
        """
        with self.lock:
            self.system_report = report
            self.mic = report.get("mic_validation", {}).get("permissions", {})
            self.service_map = report.get("services", {})

            global_status = report.get("global_status")
            if global_status == "OK":
                self.is_architecture_validated = True
                self.operational_status = "OPERATIONAL"
                logger.info(
                    "Arquitectura del sistema validada. Estado: OK. El sistema está operativo."
                )
            else:  # "ERROR" or "VIOLATION"
                self.is_architecture_validated = False
                self.operational_status = "HALTED"
                logger.error(
                    f"La validación de la arquitectura falló. Estado: {global_status}. "
                    f"El sistema se detiene por seguridad."
                )

    def _get_harmony_state(self) -> Optional[Dict[str, Any]]:
        hc_url = self.central_urls.get("harmony_controller", DEFAULT_HC_URL)
        url = f"{hc_url}/api/harmony/state"
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, timeout=REQUESTS_TIMEOUT)
                response.raise_for_status()
                json_data = response.json()
                response_data = json_data

                if (response_data.get("status") == "success" and
                        "data" in response_data):
                    data_preview = str(response_data["data"])[:100]
                    logger.debug(
                        "Estado válido recibido de Harmony: %s", data_preview
                    )
                    return response_data["data"]
                else:
                    logger.warning(
                        "Respuesta inválida desde Harmony: %s", response_data
                    )
            except Exception as e:
                logger.exception(
                    "Error inesperado (intento %s): %s", attempt + 1, e
                )

            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.debug(
                    "Reintentando obtener estado de Harmony en %.2fs...",
                    delay,
                )
                time.sleep(delay)

        logger.error(
            "No se pudo obtener estado de Harmony tras %s intentos.",
            MAX_RETRIES,
        )
        return None

    def _get_ecu_field_vector(self) -> Optional[np.ndarray]:
        """Obtiene el campo vectorial complejo completo de la ECU."""
        ecu_url = self.central_urls.get(
            "ecu_field_vector", DEFAULT_ECU_FIELD_VECTOR_URL
        )
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(ecu_url, timeout=REQUESTS_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success" and "field_vector" in data:
                    # La ECU devuelve [real, imag], necesitamos convertirlo a complejo
                    field_data = data["field_vector"]
                    complex_field = np.array(field_data, dtype=float)
                    # El array es (capas, filas, cols, 2), lo convertimos a (capas, filas, cols) de tipo complejo
                    complex_field = complex_field[..., 0] + 1j * complex_field[..., 1]
                    logger.debug("Campo vectorial complejo recibido de ECU.")
                    return complex_field
            except Exception as e:
                logger.error(
                    "Error al obtener campo vectorial de ECU (intento %s): %s",
                    attempt + 1, e
                )
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_RETRY_DELAY * (2 ** attempt))
        logger.error(
            "No se pudo obtener el campo vectorial de ECU tras %s intentos.",
            MAX_RETRIES
        )
        return None

    def calculate_coherence(self, region: np.ndarray) -> tuple[float, float]:
        """
        Calcula la coherencia y la fase dominante de una región del campo.

        Args:
            region (np.ndarray): Un array de números complejos que representa
                                 una sección del campo_q.

        Returns:
            tuple[float, float]: Una tupla conteniendo:
                - coherencia (float): Magnitud del vector de fase promedio (0 a 1).
                - fase_dominante (float): Ángulo del vector de fase promedio (en radianes).
        """
        if region.size == 0:
            return 0.0, 0.0

        # Normalizar cada vector a la fase pura (proyectar en el círculo unitario)
        # Se evita la división por cero si hay magnitudes nulas.
        magnitudes = np.abs(region)
        non_zero_magnitudes = magnitudes > 1e-9
        phases = np.zeros_like(region, dtype=np.complex128)
        phases[non_zero_magnitudes] = region[non_zero_magnitudes] / magnitudes[non_zero_magnitudes]

        # Calcular el vector de fase promedio solo sobre los elementos no nulos
        non_zero_phases = phases[non_zero_magnitudes]
        if non_zero_phases.size == 0:
            return 0.0, 0.0
        mean_phase_vector = np.mean(non_zero_phases)

        # La coherencia es la magnitud del vector promedio
        coherence = float(np.abs(mean_phase_vector))

        # La fase dominante es el ángulo del vector promedio
        dominant_phase = float(np.angle(mean_phase_vector))

        return coherence, dominant_phase

    def _delegate_phase_synchronization_task(
        self, region_identifier: str, target_phase: float
    ):
        """
        Delega una tarea de sincronización de fase a Harmony Controller.
        """
        # Comprobación de permisos basada en la MIC
        try:
            permission = self.mic["agent_ai"]["harmony_controller"]
            if permission != "CONTROL_TASK":
                logger.error(
                    f"Violación de MIC: No se tiene permiso '{permission}' para controlar harmony_controller."
                )
                return
        except KeyError:
            logger.error(
                "Violación de MIC: No hay una regla definida para agent_ai -> harmony_controller."
            )
            return

        logger.info("Permiso verificado. Delegando tarea de sincronización a harmony_controller.")

        task_url = f"{self.central_urls['harmony_controller']}/api/tasks/phase_sync"
        payload = {
            "region_identifier": region_identifier,
            "target_phase": target_phase,
            "task_type": "phase_synchronization",
        }

        logger.info(
            "Delegando tarea de sincronización para '%s' a fase %.3f rad a HC...",
            region_identifier,
            target_phase,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    task_url, json=payload, timeout=REQUESTS_TIMEOUT
                )
                response.raise_for_status()
                logger.info(
                    "Tarea de sincronización de fase delegada exitosamente a HC."
                )
                return
            except Exception as e:
                logger.error(
                    "Error al delegar tarea a HC (intento %s/%s): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    time.sleep(delay)
        logger.error(
            "No se pudo delegar la tarea de sincronización de fase a HC tras %s intentos.",
            MAX_RETRIES,
        )

    def find_resonant_frequency(self, region_identifier: str) -> Optional[float]:
        """
        Determina la frecuencia de resonancia para una región dada.

        En una implementación real, esto debería consultar a `matriz_ecu` para
        obtener el parámetro `alpha` de la capa correspondiente.
        Por ahora, simularemos este proceso. La frecuencia de resonancia es
        directamente proporcional a alpha.

        Args:
            region_identifier (str): El identificador de la región (ej. "capa_0").

        Returns:
            Optional[float]: La frecuencia de resonancia (alpha), o None si no se puede determinar.
        """
        # Simulación: En un caso real, esto sería una llamada API a la ECU.
        # Por ejemplo: `self._get_ecu_parameter(region_identifier, 'alpha')`
        # Asumimos un `alpha` de 0.5 para la capa 0 como en el default de matriz_ecu.
        if region_identifier == "capa_0":
            alpha = 0.5
            logger.info(
                "Frecuencia de resonancia (alpha) para '%s' es %.3f.",
                region_identifier,
                alpha,
            )
            return alpha
        logger.warning(
            "No se pudo determinar la frecuencia de resonancia para '%s'.",
            region_identifier,
        )
        return None

    def _delegate_resonance_task(
        self, region_identifier: str, resonant_frequency: float
    ):
        """
        Delega una tarea de excitación por resonancia a Harmony Controller.
        """
        task_url = f"{self.central_urls['harmony_controller']}/api/tasks/resonance"
        payload = {
            "region_identifier": region_identifier,
            "resonant_frequency": resonant_frequency,
            "task_type": "resonance_excitation",
        }

        logger.info(
            "Delegando tarea de resonancia para '%s' a frecuencia %.3f Hz a HC...",
            region_identifier,
            resonant_frequency,
        )

        # La lógica de envío es idéntica a la de sincronización de fase
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    task_url, json=payload, timeout=REQUESTS_TIMEOUT
                )
                response.raise_for_status()
                logger.info("Tarea de resonancia delegada exitosamente a HC.")
                return
            except Exception as e:
                logger.error(
                    "Error al delegar tarea de resonancia a HC (intento %s/%s): %s",
                    attempt + 1,
                    MAX_RETRIES,
                    e,
                )
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    time.sleep(delay)
        logger.error(
            "No se pudo delegar la tarea de resonancia a HC tras %s intentos.",
            MAX_RETRIES,
        )

    def analyze_pid_response(self, data: list, setpoint: float) -> dict:
        """
        Analiza una serie temporal de datos de respuesta de un sistema de control.
        La data se asume ordenada por timestamp.

        Args:
            data (list): Una lista de tuplas (timestamp, value).
            setpoint (float): El valor de setpoint objetivo.

        Returns:
            dict: Un diccionario con las métricas calculadas:
                  rise_time, overshoot, settling_time, oscillatory.
        """
        if not data or len(data) < 10: # Requiere suficientes datos para un análisis estable
            return {
                "rise_time": None, "overshoot": None, "settling_time": None,
                "oscillatory": False, "analysis_error": "No hay suficientes datos para el análisis."
            }

        try:
            timestamps, values = zip(*data)
            values = np.array(values)
            timestamps = np.array(timestamps) - timestamps[0] # Normalizar tiempo a 0

            # Calcular valor de estado estacionario como el promedio del último 10% de los datos
            steady_state_start_index = int(len(values) * 0.9)
            steady_state_value = np.mean(values[steady_state_start_index:])

            if steady_state_value < 1e-9:
                return {"rise_time": None, "overshoot": 0, "settling_time": None, "oscillatory": False, "analysis_error": "Valor de estado estacionario es cero."}

            # Rise Time (10% a 90% del valor de estado estacionario)
            ten_percent_val = 0.1 * steady_state_value
            ninety_percent_val = 0.9 * steady_state_value

            indices_above_10 = np.where(values >= ten_percent_val)[0]
            indices_above_90 = np.where(values >= ninety_percent_val)[0]

            rise_time = timestamps[indices_above_90[0]] - timestamps[indices_above_10[0]] if indices_above_10.any() and indices_above_90.any() else None

            # Overshoot
            peak_value = np.max(values)
            overshoot = ((peak_value - steady_state_value) / steady_state_value) * 100 if peak_value > steady_state_value else 0.0

            # Settling Time (tiempo para que la respuesta permanezca dentro del 5% del valor de estado estacionario)
            settling_band_upper = steady_state_value * 1.05
            settling_band_lower = steady_state_value * 0.95

            outside_band_indices = np.where((values > settling_band_upper) | (values < settling_band_lower))[0]

            settling_time = timestamps[outside_band_indices[-1]] if outside_band_indices.any() else timestamps[0]

            # Oscillatory (si cruza el setpoint más de 2 veces)
            crossings = np.where(np.diff(np.sign(values - setpoint)))[0]
            oscillatory = len(crossings) > 2

            return {
                "rise_time": rise_time,
                "overshoot": overshoot,
                "settling_time": settling_time,
                "oscillatory": oscillatory,
            }
        except Exception as e:
            logger.error("Error inesperado en analyze_pid_response: %s", e)
            return {
                "rise_time": None, "overshoot": None, "settling_time": None,
                "oscillatory": False, "analysis_error": str(e)
            }


    def _determine_harmony_setpoint(
        self,
        measurement,
        cogniboard_signal,
        config_status,
        strategy,
        modules,
    ) -> List[float]:
        """
        Determina el vector de setpoint objetivo para Harmony Controller.
        """
        with self.lock:
            current_target_vector = list(self.target_setpoint_vector)
            current_target_norm = np.linalg.norm(current_target_vector) \
                if current_target_vector else 0.0
            last_pid_output = self.harmony_state.get(
                "last_pid_output", 0.0
            )

        new_target_vector = list(current_target_vector)
        error_global = current_target_norm - measurement
        logger.debug(
            "[SetpointLogic] MA:%.3f, MO:%.3f, EG:%.3f",
            measurement, current_target_norm, error_global)

        aux_stats = {
            "malla": {"potenciador": 0, "reductor": 0},
            "ecu": {"potenciador": 0, "reductor": 0},
        }
        for mod_info in modules.values():
            if (
                mod_info.get("tipo") == "auxiliar"
                and mod_info.get("estado_salud") == "ok"
            ):
                aporta_a = mod_info.get("aporta_a")
                naturaleza = mod_info.get("naturaleza_auxiliar")
                if (
                    aporta_a == "malla_watcher"
                    and naturaleza in aux_stats["malla"]
                ):
                    aux_stats["malla"][naturaleza] += 1
                elif (
                    aporta_a == "matriz_ecu" and naturaleza in aux_stats["ecu"]
                ):
                    aux_stats["ecu"][naturaleza] += 1

        logger.debug("[SetpointLogic] Estrategia: %s", strategy)
        log_msg_aux = (
            "[SetpointLogic] Aux: Malla(P:%s,R:%s), ECU(P:%s,R:%s)"
        )
        logger.debug(
            log_msg_aux,
            aux_stats["malla"]["potenciador"],
            aux_stats["malla"]["reductor"],
            aux_stats["ecu"]["potenciador"],
            aux_stats["ecu"]["reductor"],
        )

        stability_threshold = (
            0.1 * current_target_norm
            if current_target_norm > 0
            else 0.1
        )
        pid_effort_threshold = 0.5

        def adjust_vector(vector, scale):
            return [x * scale for x in vector]

        if strategy == "estabilidad":
            err_low = abs(error_global) < stability_threshold
            pid_high = abs(last_pid_output) > pid_effort_threshold
            if err_low or pid_high:
                norm_vec = np.linalg.norm(new_target_vector)
                if norm_vec > 1e-6:
                    logger.info(
                        "[Estrategia Estabilidad] Reduciendo magnitud "
                        "setpoint."
                    )
                    new_target_vector = adjust_vector(
                        new_target_vector, 0.98
                    )
            if aux_stats["malla"]["reductor"] > \
                    aux_stats["malla"]["potenciador"]:
                logger.info(
                    "[Estabilidad] Más reductores en malla, reducción extra."
                )
                new_target_vector = adjust_vector(
                    new_target_vector, 0.97
                )

        elif strategy == "rendimiento":
            if (
                abs(error_global) < stability_threshold
                and abs(last_pid_output) < pid_effort_threshold / 2
            ):
                norm_vec = np.linalg.norm(new_target_vector)
                if norm_vec > 1e-6:
                    logger.info(
                        "[Estrategia Rendimiento] Aumentando magnitud "
                        "setpoint."
                    )
                    new_target_vector = adjust_vector(
                        new_target_vector, 1.02
                    )
                elif norm_vec < 1e-6:
                    logger.info(
                        "[Estrategia Rendimiento] Estableciendo setpoint "
                        "mínimo."
                    )
                    dim = len(self.target_setpoint_vector)
                    new_target_vector = [0.1] * dim
            if aux_stats["ecu"]["potenciador"] > \
                    aux_stats["ecu"]["reductor"]:
                logger.info(
                    "[Rendimiento] Más potenciadores ECU, aumento extra."
                )
                new_target_vector = adjust_vector(
                    new_target_vector, 1.01
                )

        elif strategy == "ahorro_energia":
            total_reductores = (
                aux_stats["malla"]["reductor"]
                + aux_stats["ecu"]["reductor"]
            )
            if total_reductores > 0:
                logger.info(
                    "[Ahorro Energía] Reductores activos, reducción "
                    "setpoint."
                )
                new_target_vector = adjust_vector(
                    new_target_vector, 0.95
                )

        if cogniboard_signal is not None:
            try:
                signal_val = float(cogniboard_signal)
                if signal_val > 0.8:
                    logger.info(
                        "[Cogniboard] Señal alta detectada, reduciendo "
                        "magnitud final."
                    )
                    norm = np.linalg.norm(new_target_vector)
                    if norm > 1e-6:
                        new_target_vector = adjust_vector(
                            new_target_vector, 0.9
                        )
            except (ValueError, TypeError):
                logger.warning(
                    "No se pudo convertir señal cogniboard a float: %s",
                    cogniboard_signal,
                )

        if not isinstance(new_target_vector, list):
            type_generated = type(new_target_vector)
            logger.error(
                "Error: _determine_harmony_setpoint no generó lista: %s",
                type_generated,
            )
            return list(self.target_setpoint_vector)

        return new_target_vector

    def _send_setpoint_to_harmony(self, setpoint_vector: List[float]):
        """
        Envía el setpoint vectorial calculado a Harmony Controller con
        reintentos.
        """
        hc_url = self.central_urls.get(
            "harmony_controller", DEFAULT_HC_URL
        )
        url = f"{hc_url}/api/harmony/setpoint"
        payload = {"setpoint_vector": setpoint_vector}

        logger.debug(
            "Intentando enviar setpoint a HC: %s a %s",
            setpoint_vector,
            url,
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    url, json=payload, timeout=REQUESTS_TIMEOUT)
                response.raise_for_status()
                logger.info(
                    "Setpoint %s enviado exitosamente a HC. Respuesta: %s",
                    setpoint_vector,
                    response.status_code,
                )
                return
            except Exception as e:
                err_type = type(e).__name__
                logger.error(
                    "Error al enviar setpoint a HC (%s) intento %s/%s: %s - "
                    "%s",
                    url,
                    attempt + 1,
                    MAX_RETRIES,
                    err_type,
                    e,
                )
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    logger.debug(
                        "Reintentando envío de setpoint a HC en %.2fs...",
                        delay,
                    )
                    time.sleep(delay)

        logger.error(
            "No se pudo enviar setpoint %s a HC después de %d intentos.",
            setpoint_vector,
            MAX_RETRIES,
        )

    def registrar_modulo(self, modulo_info: Dict[str, Any]) -> Dict[str, str]:
        """
        Registra un nuevo módulo watcher o tool en AgentAI.

        Valida la información proporcionada del módulo, verifica
        las dependencias si se especifica un archivo
        `requirements.txt`, y almacena los detalles del módulo.
        Inicia una validación de salud asíncrona para el módulo
        recién registrado.

        Args:
            modulo_info: Un diccionario que contiene la información
            del módulo a registrar. Se esperan claves como
            'nombre', 'url', 'url_salud',
            'tipo' ('auxiliar', 'core', etc.),
            'aporta_a' (para auxiliares),
            'naturaleza_auxiliar' (para auxiliares), y opcionalmente
            'requirements_path'.

        Returns:
            Un diccionario con el resultado de la operación,
            conteniendo una clave 'status' ('success' o 'error')
            y un 'mensaje'.
        """
        req_path = modulo_info.get("requirements_path")
        nombre = modulo_info.get("nombre")
        tipo_modulo = modulo_info.get("tipo", "desconocido")
        aporta_a = modulo_info.get("aporta_a")
        naturaleza_auxiliar = modulo_info.get("naturaleza_auxiliar")

        valido, mensaje = validate_module_registration(modulo_info)
        if not valido:
            logger.error(
                "Registro fallido para '%s' (datos inválidos): %s - Data: "
                "%s",
                nombre,
                mensaje,
                modulo_info,
            )
            return {
                "status": "error",
                "mensaje": mensaje
            }

        deps_ok = True
        deps_msg = "Validación de dependencias omitida o exitosa."
        if req_path:
            if not os.path.exists(req_path):
                deps_msg = (
                    "No se pudo encontrar el archivo de dependencias: "
                    f"{req_path}")
                logger.error(deps_msg)
                # deps_ok = False # No longer just set, return error
                return {"status": "error", "mensaje": deps_msg}
            elif not os.path.exists(GLOBAL_REQUIREMENTS_PATH):
                deps_msg = (
                    f"No se encontró el archivo"
                    f"GLOBAL_REQUIREMENTS_PATH ({GLOBAL_REQUIREMENTS_PATH})"
                    f"para validar dependencias de '%s'." % nombre
                )
                logger.error(deps_msg)
                return {"status": "error", "mensaje": deps_msg}
            else:  # Both local and global req files exist
                try:
                    deps_ok, deps_msg = check_missing_dependencies(
                        req_path, GLOBAL_REQUIREMENTS_PATH)
                    if not deps_ok:
                        logger.error(
                            "Registro fallido para '%s' (dependencias): %s",
                            nombre,
                            deps_msg,
                        )
                        return {
                            "status": "error",
                            "mensaje": deps_msg
                        }
                except Exception as e:
                    # deps_ok = False # Not needed, already returning
                    deps_msg = (
                        f"Error inesperado al verificar dependencias: {e}"
                    )
                    logger.exception(deps_msg)  # Log con stacktrace
                    return {  # Retornar error si la verificación misma falla
                        "status": "error",
                        "mensaje": deps_msg
                    }
            # This elif is no longer reachable due to the checks above,
            # can be removed or will be dead code.
            # For safety, let's assume the logic above covers all cases for req_path.
            # elif req_path and not os.path.exists(GLOBAL_REQUIREMENTS_PATH):
            #    logger.warning(
            #        "No se encontró GLOBAL_REQUIREMENTS_PATH en %s, "
            #        "omitiendo chequeo de dependencias para '%s'",
            #       GLOBAL_REQUIREMENTS_PATH,
            #        nombre,
            #   )
            #    deps_msg = "Validación omitida (archivo global no encontrado)."

        with self.lock:
            if nombre in self.modules:
                logger.warning(
                    "Intento de registrar módulo existente: %s", nombre
                )
                return {
                    "status": "error",
                    "mensaje": "El módulo ya está registrado."
                }

            module_entry = {
                "nombre": nombre,
                "url": modulo_info.get("url"),
                "url_salud": modulo_info.get("url_salud",
                                             modulo_info.get("url")),
                "tipo": tipo_modulo,
                "descripcion": modulo_info.get("descripcion", ""),
                "estado_salud": "pendiente",
                "dependencias_ok": deps_ok,
                "dependencias_msg": deps_msg,
            }
            if aporta_a:
                module_entry["aporta_a"] = aporta_a
            if tipo_modulo == "auxiliar" and naturaleza_auxiliar:
                module_entry["naturaleza_auxiliar"] = naturaleza_auxiliar

            self.modules[nombre] = module_entry

            log_details = f"Tipo: {tipo_modulo}"
            if aporta_a:
                log_details += f", Aporta a: {aporta_a}"
            if naturaleza_auxiliar:
                log_details += f", Naturaleza: {naturaleza_auxiliar}"
            logger.info(
                "Módulo '%s' (%s) registrado. %s. Pendiente de validación.",
                nombre,
                log_details,
                deps_msg)

        thread = threading.Thread(
            target=self._validar_salud_modulo,
            args=(nombre,),
            daemon=True,
            name=f"HealthCheck-{nombre}"
        )
        thread.start()
        return {
            "status": "success",
            "mensaje": f"Módulo '{nombre}' registrado"
        }

    def _validar_salud_modulo(self, nombre):
        """Valida la salud del módulo y notifica a HC si es necesario."""
        with self.lock:
            modulo = self.modules.get(nombre)
            if not modulo:
                logger.error(
                    "No se encontró el módulo '%s' para validar.",
                    nombre
                )
                return

            modulo_url_salud = modulo.get("url_salud")
            modulo_url_control = modulo.get("url")
            modulo_tipo = modulo.get("tipo")
            modulo_aporta_a = modulo.get("aporta_a")
            modulo_naturaleza = modulo.get("naturaleza_auxiliar")

        if not modulo_url_salud:
            logger.error(
                "No se encontró URL de salud para validar '%s'", nombre
            )
            estado_salud = "error_configuracion"
        else:
            estado_salud = "error_desconocido"
            for attempt in range(MAX_RETRIES):
                try:
                    logger.debug(
                        "Validando salud de '%s' en %s... (intento %d/%d)",
                        nombre,
                        modulo_url_salud,
                        attempt + 1,
                        MAX_RETRIES)
                    response = requests.get(
                        modulo_url_salud, timeout=REQUESTS_TIMEOUT
                    )
                    if response.status_code == 200:
                        estado_salud = "ok"
                        logger.info(
                            "Módulo '%s' validado (Salud OK).", nombre
                        )
                        break
                    else:
                        estado_salud = f"error_{response.status_code}"
                        logger.warning(
                            "Validación fallida para '%s'. Status: %d",
                            nombre,
                            response.status_code
                        )
                except Exception as e:
                    estado_salud = "error_inesperado"
                    logger.exception(
                        "Error inesperado al validar salud de '%s': %s",
                        nombre, e)

                if estado_salud == "ok":
                    break

                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.debug(
                        "Reintentando validación para '%s' en %.2fs...",
                        nombre,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Validación para '%s' falló tras %d intentos. "
                        "Estado: %s",
                        nombre,
                        MAX_RETRIES,
                        estado_salud
                    )

        with self.lock:
            if nombre in self.modules:
                self.modules[nombre]["estado_salud"] = estado_salud
            else:
                logger.warning(
                    "Módulo '%s' desapareció antes de actualizar estado.",
                    nombre,
                )
                return

        if (
            estado_salud == "ok"
            and modulo_tipo == "auxiliar"
            and modulo_aporta_a
            and modulo_naturaleza
        ):
            if modulo_url_control:
                logger.info(
                    "Módulo auxiliar '%s' saludable. Notificando a Harmony "
                    "Controller...",
                    nombre,
                )
                self._notify_harmony_controller_of_tool(
                    nombre=nombre,
                    url=modulo_url_control,
                    aporta_a=modulo_aporta_a,
                    naturaleza=modulo_naturaleza,
                )
            else:
                logger.error(
                    "Módulo '%s' saludable pero sin URL de control. No se "
                    "puede notificar a HC.",
                    nombre,
                )
        elif estado_salud == "ok" and modulo_tipo == "auxiliar":
            if not modulo_aporta_a:
                logger.warning(
                    "Módulo aux '%s' ok pero sin 'aporta_a'. No se "
                    "notificará.",
                    nombre,
                )
            if not modulo_naturaleza:
                logger.warning(
                    "Módulo aux '%s' ok pero sin 'naturaleza_auxiliar'. No "
                    "se notificará.",
                    nombre,
                )

    def _notify_harmony_controller_of_tool(
        self, nombre: str, url: str, aporta_a: str, naturaleza: str
    ):
        register_url = self.hc_register_url
        payload = {
            "nombre": nombre,
            "url": url,
            "aporta_a": aporta_a,
            "naturaleza": naturaleza,
        }
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(
                    "Notificando a HC sobre '%s' (Naturaleza: %s) en %s...",
                    nombre,
                    naturaleza,
                    register_url,
                )
                response = requests.post(
                    register_url, json=payload, timeout=REQUESTS_TIMEOUT)
                response.raise_for_status()
                logger.info(
                    "Notificación para '%s' enviada a HC. Respuesta: %d",
                    nombre,
                    response.status_code,
                )
                return
            except Exception as e:
                logger.exception(
                    "Error inesperado al notificar a HC: %s", e
                )
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2**attempt)
                    logger.debug(
                        "Reintentando notificación a HC para '%s' en "
                        "%.2fs...",
                        nombre,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "Fallo al notificar a HC sobre '%s' tras %d intentos.",
                        nombre,
                        MAX_RETRIES,
                    )

    def actualizar_comando_estrategico(
        self,
        comando: str,
        valor: Any
    ) -> Dict[str, str]:
        """
        Procesa y aplica comandos estratégicos de alto nivel.

        Este método permite modificar el comportamiento de AgentAI,
        como cambiar la estrategia actual o establecer directamente
        el vector de setpoint objetivo.

        Args:
            comando: El nombre del comando a ejecutar (ej: "set_strategy",
                "set_target_setpoint_vector").
            valor: El valor asociado al comando. El tipo esperado depende
                del comando específico.

        Returns:
            Un diccionario indicando el resultado de la operación, con claves
            'status' ('success' o 'error') y 'mensaje'.
        """
        with self.lock:
            if comando == "set_strategy":
                if isinstance(valor, str):
                    self.current_strategy = valor
                    logger.info("Estrategia actualizada a: %s", valor)
                    return {
                        "status": "success",
                        "mensaje": f"Estrategia establecida a '{valor}'",
                    }
                else:
                    return {
                        "status": "error",
                        "mensaje": "'set_strategy' debe ser un string",
                    }
            elif comando == "set_target_setpoint_vector":
                try:
                    if valor is None:
                        return {
                            "status": "error",
                            "mensaje": "Falta 'valor' para el comando",
                        }
                    new_vector = [float(x) for x in valor]
                    self.target_setpoint_vector = new_vector
                    logger.info(
                        "Setpoint objetivo actualizado a: %s", new_vector
                    )
                    self._send_setpoint_to_harmony(
                        self.target_setpoint_vector
                    )
                    return {
                        "status": "success",
                        "mensaje": (
                            f"Setpoint objetivo establecido a {new_vector}"
                        ),
                    }
                except (ValueError, TypeError):
                    return {
                        "status": "error",
                        "mensaje": "Valor debe ser una lista de números",
                    }
            else:
                logger.warning(
                    "Comando estratégico desconocido: %s", comando
                )
                return {
                    "status": "error",
                    "mensaje": f"Comando '{comando}' no reconocido",
                }

    def recibir_control_cogniboard(self, control_signal: Any):
        """Actualiza la señal de control recibida de Cogniboard.

        Almacena la señal para que pueda ser utilizada en la lógica de
        determinación del setpoint o en otras decisiones estratégicas.

        Args:
            control_signal: La señal recibida de Cogniboard. El tipo de dato
                puede variar según la naturaleza de la señal.
        """
        with self.lock:
            self.external_inputs["cogniboard_signal"] = control_signal
        logger.debug(
            "Señal de control de Cogniboard actualizada: %s",
            control_signal)

    def obtener_estado_completo(self) -> Dict[str, Any]:
        """
        Retorna una vista completa del estado interno actual de AgentAI.

        Este método es útil para la monitorización, depuración o para que
        otros componentes del sistema consulten el estado general de AgentAI.

        Returns:
            Un diccionario que contiene el estado actual, incluyendo el
            setpoint objetivo, la estrategia, entradas externas, el último
            estado de Harmony Controller y la lista de módulos registrados.
        """
        with self.lock:
            modules_list = [
                dict(info) for info in self.modules.values()
            ]
            harmony_state_copy = dict(self.harmony_state)
            external_inputs_copy = dict(self.external_inputs)
            target_setpoint_copy = list(self.target_setpoint_vector)
            current_strategy_copy = self.current_strategy

        return {
            "target_setpoint_vector": target_setpoint_copy,
            "current_strategy": current_strategy_copy,
            "external_inputs": external_inputs_copy,
            "harmony_controller_last_state": harmony_state_copy,
            "registered_modules": modules_list,
        }


def strategic_loop(agent_instance: AgentAI):
    """
    El bucle principal que ejecuta la lógica de negocio de AgentAI.
    Se ejecuta en un hilo separado.
    """
    logger.info("Iniciando bucle estratégico...")
    while True:
        start_time = time.monotonic()

        # --- Puerta de Seguridad ---
        # No operar si la arquitectura no es válida.
        with agent_instance.lock:
            is_validated = agent_instance.is_architecture_validated
            current_status = agent_instance.operational_status

        if not is_validated:
            logger.warning(
                f"La arquitectura no está validada. Estado actual: {current_status}. "
                f"Pausando bucle estratégico durante 60s."
            )
            time.sleep(60)
            continue

        try:
            state = agent_instance._get_harmony_state()
            if state is None:
                logger.error(
                    "No se pudo obtener el estado de Harmony. "
                    "El bucle estratégico esperará 30 segundos antes de reintentar."
                )
                time.sleep(30)
                continue

            current_harmony_state = state
            with agent_instance.lock:
                agent_instance.harmony_state = current_harmony_state

            with agent_instance.lock:
                current_measurement = agent_instance.harmony_state.get(
                    "last_measurement", 0.0
                )
                cogniboard_signal = agent_instance.external_inputs[
                    "cogniboard_signal"
                ]
                config_status = agent_instance.external_inputs["config_status"]
                strategy = agent_instance.current_strategy
                modules_copy = dict(agent_instance.modules)

                new_setpoint_vector = agent_instance._determine_harmony_setpoint(
                    current_measurement,
                    cogniboard_signal,
                    config_status,
                    strategy,
                    modules_copy,
                )
            with agent_instance.lock:
                setpoint_changed = not np.allclose(
                    agent_instance.target_setpoint_vector,
                    new_setpoint_vector,
                    rtol=1e-5,
                    atol=1e-8,
                )
                if setpoint_changed:
                    agent_instance.target_setpoint_vector = new_setpoint_vector
                    logger.info(
                        "Nuevo setpoint estratégico determinado: %s",
                        [
                            f"{x:.3f}"
                            for x in agent_instance.target_setpoint_vector
                        ],
                    )

            if setpoint_changed:
                agent_instance._send_setpoint_to_harmony(agent_instance.target_setpoint_vector)

            field_vector = agent_instance._get_ecu_field_vector()
            if field_vector is not None:
                coherence, dominant_phase = agent_instance.calculate_coherence(
                    field_vector[0]
                )
                logger.info(
                    "Coherencia Capa 0: %.3f, Fase Dominante: %.3f rad",
                    coherencia,
                    dominant_phase,
                )

                if coherence < 0.8:
                    logger.warning(
                        "Coherencia (%.3f) por debajo del umbral. "
                        "Iniciando maniobra de sincronización de fase.",
                        coherencia,
                    )
                    agent_instance._delegate_phase_synchronization_task(
                        region_identifier="capa_0", target_phase=dominant_phase
                    )
                elif coherence > 0.95:
                    logger.info(
                        "Coherencia alta (%.3f). Intentando maniobra de resonancia.",
                        coherencia,
                    )
                    resonant_frequency = agent_instance.find_resonant_frequency("capa_0")
                    if resonant_frequency is not None:
                        agent_instance._delegate_resonance_task("capa_0", resonant_frequency)

        except Exception as e:
            logger.exception("Error inesperado en el bucle estratégico: %s", e)

        elapsed_time = time.monotonic() - start_time
        sleep_time = max(0, STRATEGIC_LOOP_INTERVAL - elapsed_time)
        time.sleep(sleep_time)


# --- Instancia Global ---
# Esta instancia será importada por otros módulos (como endpoints.py)
# para acceder a la lógica de AgentAI.
agent_ai_instance = AgentAI()

# El hilo del bucle estratégico se inicia ahora desde el punto de entrada
# principal de la aplicación (en endpoints.py) para asegurar que solo se
# inicie una vez.
_strategic_thread = None

def start_loop():
    """Inicia el bucle estratégico si no está ya en ejecución."""
    global _strategic_thread
    if _strategic_thread is None or not _strategic_thread.is_alive():
        logger.info("Iniciando el bucle estratégico de AgentAI...")
        _strategic_thread = threading.Thread(
            target=strategic_loop,
            args=(agent_ai_instance,),
            daemon=True,
            name="AgentAIStrategicLoop"
        )
        _strategic_thread.start()
    else:
        logger.info("El bucle estratégico de AgentAI ya está en ejecución.")

def shutdown():
    """Función de limpieza (actualmente un placeholder)."""
    logger.info("AgentAI shutdown.")
    # En un futuro, aquí se podrían añadir lógicas de apagado seguro.
    pass
