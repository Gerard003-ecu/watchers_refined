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
from flask import Flask, jsonify, request
import numpy as np
import requests

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
AGENT_AI_MALLA_URL_ENV = "AGENT_AI_MALLA_URL"
DEFAULT_HC_URL = "http://harmony_controller:7000"
DEFAULT_ECU_URL = "http://ecu:8000"
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
        self._stop_event = threading.Event()

        self.central_urls: Dict[str, str] = {}
        hc_url = os.environ.get(HARMONY_CONTROLLER_URL_ENV)
        if hc_url:
            hc_url_val = hc_url
        else:
            hc_url_val = DEFAULT_HC_URL
        self.central_urls["harmony_controller"] = hc_url_val

        ecu_url = os.environ.get(AGENT_AI_ECU_URL_ENV)
        self.central_urls["ecu"] = ecu_url if ecu_url else DEFAULT_ECU_URL

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

        self._strategic_thread = threading.Thread(
            target=self._strategic_loop,
            daemon=True,
            name="AgentAIStrategicLoop",
        )
        logger.info("AgentAI inicializado.")

    def start_loop(self):
        """Inicia el bucle estratégico principal de AgentAI.

        Este bucle se ejecuta en un hilo separado y es responsable de
        monitorizar continuamente el estado del sistema, determinar y enviar
        setpoints al Harmony Controller, y reaccionar a cambios y comandos.
        Si el bucle ya está en ejecución, esta función no hace nada.
        """
        if not self._strategic_thread.is_alive():
            self._stop_event.clear()
            self._strategic_thread = threading.Thread(
                target=self._strategic_loop,
                daemon=True,
                name="AgentAIStrategicLoop",
            )
            self._strategic_thread.start()
            logger.info("Bucle estratégico iniciado.")
        else:
            logger.info("Bucle estratégico ya está corriendo.")

    def _strategic_loop(self):
        """Bucle principal que ejecuta la lógica estratégica periódicamente."""
        logger.info(
            "Esperando a que harmony_controller esté disponible..."
        )
        max_retries_hc_wait = 30
        retry_interval_hc_wait = 2
        hc_ready = False
        for attempt in range(max_retries_hc_wait):
            try:
                url = f"{HARMONY_CONTROLLER_URL}/api/harmony/state"
                response = requests.get(url, timeout=REQUESTS_TIMEOUT)
                if response.status_code == 200:
                    logger.info(
                        "Conexión establecida con harmony_controller"
                    )
                    hc_ready = True
                    break
            except Exception as e:
                logger.debug(
                    "Intento %s/%s: HC no disponible (%s)",
                    attempt + 1,
                    max_retries_hc_wait,
                    str(e),
                )
                time.sleep(retry_interval_hc_wait)
        if not hc_ready:
            logger.warning(
                "No existe conexión inicial con harmony_controller."
            )

        logger.info("Iniciando bucle estratégico...")
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            try:
                state = self._get_harmony_state()
                current_harmony_state = state
                with self.lock:
                    if current_harmony_state is not None:
                        self.harmony_state = current_harmony_state

                with self.lock:
                    current_measurement = self.harmony_state.get(
                        "last_measurement", 0.0
                    )
                    cogniboard_signal = self.external_inputs[
                        "cogniboard_signal"
                    ]
                    config_status = self.external_inputs["config_status"]
                    strategy = self.current_strategy
                    modules_copy = dict(self.modules)

                    new_setpoint_vector = self._determine_harmony_setpoint(
                        current_measurement,
                        cogniboard_signal,
                        config_status,
                        strategy,
                        modules_copy,
                    )
                with self.lock:
                    setpoint_changed = not np.allclose(
                        self.target_setpoint_vector,
                        new_setpoint_vector,
                        rtol=1e-5,
                        atol=1e-8,
                    )
                    if setpoint_changed:
                        self.target_setpoint_vector = new_setpoint_vector
                        logger.info(
                            "Nuevo setpoint estratégico determinado: %s",
                            [
                                f"{x:.3f}"
                                for x in self.target_setpoint_vector
                            ],
                        )

                if setpoint_changed:
                    self._send_setpoint_to_harmony(self.target_setpoint_vector)

            except BaseException as e:
                if isinstance(e, (SystemExit, KeyboardInterrupt)):
                    logger.info(
                        "Señal de salida recibida en bucle estratégico.")
                    break
                logger.exception("Error inesperado en el bucle estratégico.")

            elapsed_time = time.monotonic() - start_time
            sleep_time = max(0, STRATEGIC_LOOP_INTERVAL - elapsed_time)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)
        logger.info("Bucle estratégico detenido.")

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

    def _determine_harmony_setpoint(
        self,
        measurement,
        cogniboard_signal,
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
                deps_ok = False
            elif os.path.exists(GLOBAL_REQUIREMENTS_PATH):
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
                    deps_ok = False
                    deps_msg = (
                        f"Error inesperado al verificar dependencias: {e}"
                    )
                logger.exception(deps_msg)
                return {
                    "status": "error",
                    "mensaje": deps_msg
                }
            else:
                logger.warning(
                    "No se encontró GLOBAL_REQUIREMENTS_PATH en %s, "
                    "omitiendo chequeo de dependencias para '%s'",
                    GLOBAL_REQUIREMENTS_PATH,
                    nombre,
                )
                deps_msg = "Validación omitida (archivo global no encontrado)."

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

    def recibir_config_status(self, config_status: Any):
        """
        Actualiza el estado de configuración recibido de Config Agent.

        Almacena el estado de configuración para que pueda ser utilizado
        en la lógica de determinación del setpoint o en otras decisiones
        estratégicas.

        Args:
            config_status: El estado de configuración recibido.
            El tipo de dato puede variar.
        """
        with self.lock:
            self.external_inputs["config_status"] = config_status
        logger.debug(
            "Estado de configuración actualizado: %s", config_status)

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

    def shutdown(self):
        """Detiene el bucle estratégico principal de AgentAI de forma segura.

        Señala al hilo del bucle estratégico que debe detenerse y espera a que
        finalice.
        """
        logger.info("Solicitando detención del bucle estratégico...")
        self._stop_event.set()
        if self._strategic_thread.is_alive():
            self._strategic_thread.join(
                timeout=STRATEGIC_LOOP_INTERVAL + 1
            )
            if self._strategic_thread.is_alive():
                logger.warning(
                    "El hilo estratégico no terminó limpiamente."
                )


# --- Instancia Global ---
agent_ai_instance_app = AgentAI()
agent_api = Flask(__name__)


@agent_api.route("/api/register", methods=["POST"])
def handle_register():
    """Punto de entrada de API para registrar un nuevo módulo.

    Recibe datos JSON con la información del módulo, los pasa a la instancia
    de AgentAI para su procesamiento y retorna el resultado.

    Returns:
        Una respuesta JSON con el estado del registro y un código HTTP.
    """
    data = request.get_json()
    if not data:
        return (
            jsonify(
                {"status": "error", "message": "No JSON data received"}
            ),
            400,
        )
    result = agent_ai_instance_app.registrar_modulo(data)
    status_code = 200 if result.get("status") == "success" else 400
    return jsonify(result), status_code


@agent_api.route("/api/state", methods=["GET"])
def handle_get_state():
    """Punto de entrada de API para obtener el estado completo de AgentAI.

    Retorna una representación JSON del estado interno actual de la instancia
    de AgentAI.

    Returns:
        Una respuesta JSON con el estado completo y código HTTP 200.
    """
    return jsonify(agent_ai_instance_app.obtener_estado_completo()), 200


@agent_api.route("/api/command", methods=["POST"])
def handle_command():
    """Punto de entrada de API para enviar comandos estratégicos a AgentAI.

    Recibe un comando y su valor en formato JSON, los procesa a través de
    la instancia de AgentAI y retorna el resultado.

    Returns:
        Una respuesta JSON con el estado del comando y un código HTTP.
    """
    data = request.get_json()
    if not data or "comando" not in data or "valor" not in data:
        return jsonify({"status": "error", "message": "Comando inválido"}), 400
    result = agent_ai_instance_app.actualizar_comando_estrategico(
        data["comando"], data["valor"]
    )
    status_code = 200 if result.get("status") == "success" else 400
    return jsonify(result), status_code


def run_agent_ai_service():
    """Ejecuta el servicio Flask de AgentAI.

    Inicializa el bucle estratégico de AgentAI y luego inicia el servidor
    Flask para atender las peticiones API.
    """
    port = int(os.environ.get("AGENT_AI_PORT", 9000))
    logger.info("Iniciando servicio Flask para AgentAI en puerto %d...", port)
    agent_ai_instance_app.start_loop()
    agent_api.run(
        host="0.0.0.0", port=port, debug=False, use_reloader=False
    )


if __name__ == "__main__":
    try:
        run_agent_ai_service()
    except KeyboardInterrupt:
        logger.info("Interrupción manual.")
    finally:
        agent_ai_instance_app.shutdown()
        logger.info("AgentAI finalizado.")
