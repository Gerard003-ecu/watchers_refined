#!/usr/bin/env python3
"""
agent_ai.py - Núcleo Estratégico del Ecosistema Watchers

Orquesta el sistema a alto nivel:
- Gestiona el registro y salud de módulos (watchers_tools), capturando afinidad ('aporta_a') y naturaleza ('naturaleza_auxiliar').
- Notifica a Harmony Controller sobre módulos auxiliares saludables, su afinidad y naturaleza.
- Monitoriza el estado general a través de harmony_controller.
- Determina el estado deseado (setpoint de armonía).
- Comunica el setpoint a harmony_controller.
- Procesa entradas externas (cogniboard, config_agent) y comandos estratégicos.
"""


from flask import Flask, request, jsonify
import threading
import time
import requests
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from validation.validator import (
    validate_module_registration,
    check_missing_dependencies
)
from utils.logger import get_logger

logger = get_logger()

# --- Configuración ---
HARMONY_CONTROLLER_URL = os.environ.get(
    "HARMONY_CONTROLLER_URL",
    "http://harmony_controller:7000")
HARMONY_CONTROLLER_REGISTER_URL = os.environ.get(
    "HARMONY_CONTROLLER_REGISTER_URL",
    f"{HARMONY_CONTROLLER_URL}/api/harmony/register_tool")
HARMONY_CONTROLLER_URL_ENV = "HARMONY_CONTROLLER_URL"  #  Nombre de la variable ENV
HARMONY_CONTROLLER_REGISTER_URL_ENV = "HARMONY_CONTROLLER_REGISTER_URL"
# --- NUEVO: Variables de Entorno para Centrales Esenciales ---
AGENT_AI_ECU_URL_ENV = "AGENT_AI_ECU_URL"
AGENT_AI_MALLA_URL_ENV = "AGENT_AI_MALLA_URL"
# --- NUEVO: Defaults si ENV no está definida ---
DEFAULT_HC_URL = "http://harmony_controller:7000"
DEFAULT_ECU_URL = "http://ecu:8000"
DEFAULT_MALLA_URL = "http://malla_watcher:5001"
# --- FIN NUEVO ---
STRATEGIC_LOOP_INTERVAL = float(os.environ.get('AA_INTERVAL', 5.0))
REQUESTS_TIMEOUT = float(os.environ.get('AA_REQUESTS_TIMEOUT', 4.0))
GLOBAL_REQUIREMENTS_PATH = os.environ.get(
    'AA_GLOBAL_REQ_PATH', "/app/requirements.txt")
MAX_RETRIES = int(os.environ.get('AA_MAX_RETRIES', 3))
BASE_RETRY_DELAY = float(os.environ.get('AA_BASE_RETRY_DELAY', 0.5))


class AgentAI:
    def __init__(self):
        self.modules: Dict[str, Dict] = {}
        self.harmony_state: Dict[str, Any] = {}
        try:
            initial_vector_str = os.environ.get(
                'AA_INITIAL_SETPOINT_VECTOR', '[1.0, 0.0]')
            parsed_vector = json.loads(initial_vector_str)
            if isinstance(
                parsed_vector, list) and all(
                isinstance(
                    x, (int, float)) for x in parsed_vector):
                self.target_setpoint_vector: List[float] = parsed_vector
            else:
                raise ValueError(
                    "El valor parseado no es una lista de números")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                f"AA_INITIAL_SETPOINT_VECTOR ('{initial_vector_str}') inválido ({e}), usando default [1.0, 0.0]")
            self.target_setpoint_vector = [1.0, 0.0]
        self.current_strategy: str = os.environ.get(
            'AA_INITIAL_STRATEGY', 'default')
        self.external_inputs: Dict[str, Any] = {
            "cogniboard_signal": None,
            "config_status": None
        }
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        # --- NUEVO: Leer y almacenar URLs de Centrales Esenciales ---
        self.central_urls: Dict[str, str] = {}  #  Inicializar diccionario
        # Leer URLs manejando explícitamente None o ""
        hc_url = os.environ.get(HARMONY_CONTROLLER_URL_ENV)
        # Usar default si es None o ""
        self.central_urls["harmony_controller"] = hc_url if hc_url else DEFAULT_HC_URL

        ecu_url = os.environ.get(AGENT_AI_ECU_URL_ENV)
        self.central_urls["ecu"] = ecu_url if ecu_url else DEFAULT_ECU_URL

        malla_url = os.environ.get(AGENT_AI_MALLA_URL_ENV)
        self.central_urls["malla_watcher"] = malla_url if malla_url else DEFAULT_MALLA_URL

        logger.info(
            f"URLs Centrales configuradas: HC='{self.central_urls['harmony_controller']}', "
            f"ECU='{self.central_urls['ecu']}', Malla='{self.central_urls['malla_watcher']}'")
        # Leer URL de registro manejando explícitamente None o ""
        hc_reg_url_env = os.environ.get(HARMONY_CONTROLLER_REGISTER_URL_ENV)
        # Leer URL de registro manejando explícitamente None o ""
        self.hc_register_url = hc_reg_url_env if hc_reg_url_env else \
            f"{self.central_urls['harmony_controller']}/api/harmony/register_tool"
        logger.info(
            f"URL de registro de Harmony Controller: '{self.hc_register_url}'")
        # --- FIN NUEVO ---
        self._strategic_thread = threading.Thread(
            target=self._strategic_loop, daemon=True, name="AgentAIStrategicLoop")
        logger.info("AgentAI inicializado.")

    def start_loop(self):
        """Inicia el bucle estratégico si no está corriendo."""
        if not self._strategic_thread.is_alive():
            self._stop_event.clear()
            self._strategic_thread = threading.Thread(
                target=self._strategic_loop, daemon=True, name="AgentAIStrategicLoop")
            self._strategic_thread.start()
            logger.info("Bucle estratégico iniciado.")
        else:
            logger.info("Bucle estratégico ya está corriendo.")

    def _strategic_loop(self):
        """Bucle principal que ejecuta la lógica estratégica periódicamente."""
        logger.info("Esperando a que harmony_controller esté disponible...")
        max_retries_hc_wait = 30
        retry_interval_hc_wait = 2
        hc_ready = False
        for attempt in range(max_retries_hc_wait):
            try:
                response = requests.get(
                    f"{HARMONY_CONTROLLER_URL}/api/harmony/state",
                    timeout=REQUESTS_TIMEOUT)
                if response.status_code == 200:
                    logger.info("Conexión establecida con harmony_controller")
                    hc_ready = True
                    break
            except Exception as e:
                logger.debug(
                    f"Intento {attempt+1}/{max_retries_hc_wait}: harmony_controller no disponible ({str(e)})")
                time.sleep(retry_interval_hc_wait)
        if not hc_ready:
            logger.warning(
                "No se pudo establecer conexión inicial con harmony_controller.")

        logger.info("Iniciando bucle estratégico...")
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            try:
                current_harmony_state = self._get_harmony_state()
                with self.lock:
                    if current_harmony_state is not None:
                        self.harmony_state = current_harmony_state

                with self.lock:
                    current_measurement = self.harmony_state.get(
                        "last_measurement", 0.0)
                    cogniboard_signal = self.external_inputs["cogniboard_signal"]
                    config_status = self.external_inputs["config_status"]
                    strategy = self.current_strategy
                    modules_copy = dict(self.modules)

                    new_setpoint_vector = self._determine_harmony_setpoint(
                        current_measurement, cogniboard_signal, config_status, strategy, modules_copy)
                with self.lock:
                    setpoint_changed = not np.allclose(
                        self.target_setpoint_vector, new_setpoint_vector,
                        rtol=1e-5, atol=1e-8)
                    if setpoint_changed:
                        self.target_setpoint_vector = new_setpoint_vector
                        logger.info(
                            "Nuevo setpoint estratégico determinado: "
                            f"{[f'{x:.3f}' for x in self.target_setpoint_vector]}")

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

        # ... (_get_harmony_state, _determine_harmony_setpoint, _send_setpoint_to_harmony sin cambios funcionales) ...
    def _get_harmony_state(self) -> Optional[Dict[str, Any]]:
        # Usar la URL almacenada
        url = f"{self.central_urls.get('harmony_controller', DEFAULT_HC_URL)}/api/harmony/state"
        # --- INICIO BLOQUE INDENTADO ---
        for attempt in range(MAX_RETRIES):
            response_data = None
            try:
                response = requests.get(url, timeout=REQUESTS_TIMEOUT)
                response.raise_for_status()
                response_data = response.json()

                if response_data.get(
                        "status") == "success" and "data" in response_data:
                    logger.debug(
                        f"Estado válido recibido de Harmony: {response_data['data']}")
                    #  <-- Correcto: dentro de if/try/for/def
                    return response_data["data"]
                else:
                    logger.warning(
                        f"Respuesta inválida desde Harmony: {response_data}")

            # --- Orden de Excepts Revisado (o simplificado a Exception) ---
            except Exception as e:
                logger.exception(
                    f"Error inesperado (intento {attempt+1}): {e}")
                #  estado_salud = "error_inesperado"
            # --- Fin Excepts ---

            # --- Lógica de reintento ---
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (2 ** attempt)
                logger.debug(
                    f"Reintentando obtener estado de Harmony en {delay:.2f}s...")
                time.sleep(delay)
        # --- FIN BLOQUE INDENTADO ---

        # Si el bucle termina sin éxito
        logger.error(
            f"No se pudo obtener estado válido de Harmony después de {MAX_RETRIES} intentos.")
        return None  #  <-- Correcto: return final fuera del bucle, dentro de la función

        # --- Lógica de reintento (sin cambios) ---
        if attempt < MAX_RETRIES - 1:
            delay = BASE_RETRY_DELAY * (2 ** attempt)
            logger.debug(
                f"Reintentando obtener estado de Harmony en {delay:.2f}s...")
            time.sleep(delay)

        # logger.error(...)
        # return None

    def _determine_harmony_setpoint(
            self,
            measurement,
            cogniboard_signal,
            config_status,
            strategy,
            modules) -> List[float]:
        """
        Determina el vector de setpoint objetivo para Harmony Controller
        basado en el estado actual, la estrategia y la composición de módulos.
        Refinado para facilitar extensión y ajuste dinámico.
        """
        with self.lock:
            current_target_vector = list(self.target_setpoint_vector)
            current_target_norm = np.linalg.norm(
                current_target_vector) if current_target_vector else 0.0
            last_pid_output = self.harmony_state.get('last_pid_output', 0.0)

        new_target_vector = list(current_target_vector)
        error_global = current_target_norm - measurement
        logger.debug(
            f"[SetpointLogic] Norma Actual: {measurement:.3f}, Norma Objetivo: {current_target_norm:.3f}, Error Global: {error_global:.3f}")

        # Analizar composición de auxiliares activos
        aux_stats = {
            "malla": {"potenciador": 0, "reductor": 0},
            "ecu": {"potenciador": 0, "reductor": 0}
        }
        for mod_info in modules.values():
            if mod_info.get("tipo") == "auxiliar" and mod_info.get(
                    "estado_salud") == "ok":
                aporta_a = mod_info.get("aporta_a")
                naturaleza = mod_info.get("naturaleza_auxiliar")
                if aporta_a == "malla_watcher" and naturaleza in aux_stats["malla"]:
                    aux_stats["malla"][naturaleza] += 1
                elif aporta_a == "matriz_ecu" and naturaleza in aux_stats["ecu"]:
                    aux_stats["ecu"][naturaleza] += 1

        logger.debug(f"[SetpointLogic] Estrategia: {strategy}")
        logger.debug(
            f"[SetpointLogic] Aux Activos - Malla(P:{aux_stats['malla']['potenciador']}, "
            f"R:{aux_stats['malla']['reductor']}), ECU(P:{aux_stats['ecu']['potenciador']}, "
            f"R:{aux_stats['ecu']['reductor']})")

        stability_threshold = 0.1 * current_target_norm if current_target_norm > 0 else 0.1
        pid_effort_threshold = 0.5

        def adjust_vector(vector, scale):
            return [x * scale for x in vector]

        if strategy == "estabilidad":
            if abs(error_global) < stability_threshold or abs(
                    last_pid_output) > pid_effort_threshold:
                norm_vec = np.linalg.norm(new_target_vector)
                if norm_vec > 1e-6:
                    logger.info(
                        "[Estrategia Estabilidad] Reduciendo magnitud setpoint.")
                    new_target_vector = adjust_vector(new_target_vector, 0.98)
            # Ajuste por desbalance de potenciadores/reductores (opcional)
            # Ejemplo: Si hay más reductores que potenciadores, reducir aún más
            if aux_stats["malla"]["reductor"] > aux_stats["malla"]["potenciador"]:
                logger.info(
                    "[Estabilidad] Más reductores en malla, reducción extra.")
                new_target_vector = adjust_vector(new_target_vector, 0.97)

        elif strategy == "rendimiento":
            if abs(error_global) < stability_threshold and abs(
                    last_pid_output) < pid_effort_threshold / 2:
                norm_vec = np.linalg.norm(new_target_vector)
                if norm_vec > 1e-6:
                    logger.info(
                        "[Estrategia Rendimiento] Aumentando magnitud setpoint.")
                    new_target_vector = adjust_vector(new_target_vector, 1.02)
                elif norm_vec < 1e-6:
                    logger.info(
                        "[Estrategia Rendimiento] Estableciendo setpoint mínimo.")
                    dim = len(self.target_setpoint_vector)
                    new_target_vector = [0.1] * dim
            # Ajuste por potenciadores activos
            if aux_stats["ecu"]["potenciador"] > aux_stats["ecu"]["reductor"]:
                logger.info(
                    "[Rendimiento] Más potenciadores en ECU, aumento extra.")
                new_target_vector = adjust_vector(new_target_vector, 1.01)

        elif strategy == "ahorro_energia":
            # Ejemplo: reducir setpoint si hay muchos reductores activos
            total_reductores = aux_stats["malla"]["reductor"] + \
                aux_stats["ecu"]["reductor"]
            if total_reductores > 0:
                logger.info(
                    "[Ahorro Energía] Reductores activos, reducción de setpoint.")
                new_target_vector = adjust_vector(new_target_vector, 0.95)

        # Estrategia por defecto: no hacer nada extra

        # Ajuste final basado en Cogniboard
        if cogniboard_signal is not None:
            try:
                signal_val = float(cogniboard_signal)
                if signal_val > 0.8:
                    logger.info(
                        "[Cogniboard] Señal alta detectada, reduciendo magnitud final.")
                    norm = np.linalg.norm(new_target_vector)
                    if norm > 1e-6:
                        new_target_vector = adjust_vector(
                            new_target_vector, 0.9)
            except (ValueError, TypeError):
                logger.warning(
                    f"No se pudo convertir la señal de cogniboard a float: {cogniboard_signal}")

        if not isinstance(new_target_vector, list):
            logger.error(
                f"Error interno: _determine_harmony_setpoint no generó una lista: {type(new_target_vector)}")
            return list(self.target_setpoint_vector)

        return new_target_vector

    def _send_setpoint_to_harmony(self, setpoint_vector: List[float]):
        """
        Envía el setpoint vectorial calculado a Harmony Controller con reintentos.
        """
        # Usar la URL almacenada leída desde ENV o default
        url = f"{self.central_urls.get('harmony_controller', DEFAULT_HC_URL)}/api/harmony/setpoint"
        payload = {"setpoint_vector": setpoint_vector}
        # Log inicial
        logger.debug(
            f"Intentando enviar setpoint a HC: {setpoint_vector} a {url}")

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    url, json=payload, timeout=REQUESTS_TIMEOUT)
                response.raise_for_status()  #  Lanza excepción para errores 4xx/5xx
                logger.info(
                    f"Setpoint {setpoint_vector} enviado exitosamente a HC. Respuesta: {response.status_code}")
                return  #  Salir de la función si el envío es exitoso

            except Exception as e:  #  Captura simplificada de cualquier excepción
                logger.error(
                    f"Error al enviar setpoint a HC ({url}) intento {attempt+1}/{MAX_RETRIES}: {type(e).__name__} - {e}")
                # Lógica de espera para reintento (DENTRO DEL BUCLE)
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.debug(
                        f"Reintentando envío de setpoint a HC en {delay:.2f}s...")
                    time.sleep(delay)  #  Esperar antes del siguiente intento
                # else: #  No es necesario un else aquí, el log de fallo final
                # va después del bucle

        # Si el bucle termina sin un 'return' exitoso:
        logger.error(
            f"No se pudo enviar setpoint {setpoint_vector} a HC después de {MAX_RETRIES} intentos.")

    ### MODIFICADO: registrar_modulo ahora almacena 'naturaleza_auxiliar' ###
    def registrar_modulo(self, modulo_info):
        """
        Registra un nuevo módulo, valida dependencias y almacena afinidad y naturaleza.
        Inicia la validación de salud asíncrona.
        """
        req_path = modulo_info.get("requirements_path")
        nombre = modulo_info.get("nombre")
        tipo_modulo = modulo_info.get("tipo", "desconocido")
        aporta_a = modulo_info.get("aporta_a")
        ### NUEVO: Obtener naturaleza auxiliar ###
        naturaleza_auxiliar = modulo_info.get(
            "naturaleza_auxiliar")  #  Puede ser None

        # --- Validación de Datos de Registro ---
        valido, mensaje = validate_module_registration(modulo_info)
        if not valido:
            logger.error(
                f"Registro fallido para '{nombre}' (datos inválidos): {mensaje} - Data: {modulo_info}")
            return {"status": "error", "mensaje": mensaje}

        # --- Validación de Dependencias (sin cambios) ---
        deps_ok = True
        deps_msg = "Validación de dependencias omitida o exitosa."
        # ... (código de validación de dependencias sin cambios) ...
        if req_path:
            if not os.path.exists(req_path):
                deps_msg = f"No se pudo encontrar el archivo de dependencias del módulo: {req_path}"
                logger.error(deps_msg)
                deps_ok = False
            elif os.path.exists(GLOBAL_REQUIREMENTS_PATH):
                try:
                    deps_ok, deps_msg = check_missing_dependencies(
                        req_path, GLOBAL_REQUIREMENTS_PATH)
                    if not deps_ok:
                        logger.error(
                            f"Registro fallido para '{nombre}' (dependencias): {deps_msg}")
                        return {"status": "error", "mensaje": deps_msg}
                except Exception as e:
                    deps_ok = False
                    deps_msg = f"Error inesperado al verificar dependencias: {e}"
                    logger.exception(deps_msg)
                    return {"status": "error", "mensaje": deps_msg}
            else:
                logger.warning(
                    f"No se encontró GLOBAL_REQUIREMENTS_PATH en {GLOBAL_REQUIREMENTS_PATH}, omitiendo chequeo de dependencias para '{nombre}'")
                deps_msg = "Validación de dependencias omitida (archivo global no encontrado)."

        # --- Almacenamiento del Módulo (incluyendo aporta_a y naturaleza_auxiliar) ---
        with self.lock:
            if nombre in self.modules:
                logger.warning(
                    f"Intento de registrar módulo existente: {nombre}")
                return {
                    "status": "error",
                    "mensaje": "El módulo ya está registrado."}

            module_entry = {
                "nombre": nombre,
                "url": modulo_info.get("url"),
                "url_salud": modulo_info.get(
                    "url_salud",
                    modulo_info.get("url")),
                "tipo": tipo_modulo,
                "descripcion": modulo_info.get(
                    "descripcion",
                    ""),
                "estado_salud": "pendiente",
                "dependencias_ok": deps_ok,
                "dependencias_msg": deps_msg,
            }
            # Añadir campos específicos si existen
            if aporta_a:
                module_entry["aporta_a"] = aporta_a
            ### NUEVO: Añadir naturaleza si es auxiliar y existe ###
            if tipo_modulo == "auxiliar" and naturaleza_auxiliar:
                module_entry["naturaleza_auxiliar"] = naturaleza_auxiliar

            self.modules[nombre] = module_entry

            # Logging mejorado
            log_details = f"Tipo: {tipo_modulo}"
            if aporta_a:
                log_details += f", Aporta a: {aporta_a}"
            if naturaleza_auxiliar:
                log_details += f", Naturaleza: {naturaleza_auxiliar}"
            logger.info(
                f"Módulo '{nombre}' ({log_details}) registrado. {deps_msg}. Pendiente de validación de salud.")

        # Iniciar validación de salud
        threading.Thread(target=self._validar_salud_modulo, args=(
            nombre,), daemon=True, name=f"HealthCheck-{nombre}").start()
        return {
            "status": "success",
            "mensaje": f"Módulo '{nombre}' registrado"}

    # MODIFICADO: _validar_salud_modulo ahora obtiene y pasa
    # 'naturaleza_auxiliar ###
    def _validar_salud_modulo(self, nombre):
        """Valida la conectividad/salud inicial y notifica a HC si es auxiliar."""

        # --- Inicializaciones ---
        modulo_url_salud = None
        modulo_url_control = None
        modulo_tipo = None
        modulo_aporta_a = None
        modulo_naturaleza = None
        estado_salud = "error_inicial"  #  Estado inicial por defecto

        with self.lock:
            modulo = self.modules.get(nombre)
            if not modulo:
                logger.error(
                    f"No se encontró el módulo '{nombre}' para validar salud (ya eliminado?).")
                return
            modulo_url_salud = modulo.get("url_salud")
            modulo_url_control = modulo.get("url")
            modulo_tipo = modulo.get("tipo")
            modulo_aporta_a = modulo.get("aporta_a")
            modulo_naturaleza = modulo.get("naturaleza_auxiliar")

        # --- Comprobar URL y Realizar Validación ---
        if not modulo_url_salud:
            logger.error(
                f"No se encontró URL de salud para validar módulo '{nombre}'")
            estado_salud = "error_configuracion"  #  Asignar estado final aquí
        else:
            # --- INICIO: Lógica de Validación (DENTRO DEL ELSE) ---
            estado_salud = "error_desconocido"  #  Estado inicial para el bucle
            for attempt in range(MAX_RETRIES):
                try:
                    logger.debug(
                        f"Validando salud de '{nombre}' en {modulo_url_salud}... (intento {attempt+1}/{MAX_RETRIES})")
                    response = requests.get(
                        modulo_url_salud, timeout=REQUESTS_TIMEOUT)

                    if response.status_code == 200:
                        estado_salud = "ok"
                        logger.info(
                            f"Módulo '{nombre}' validado exitosamente (Salud OK).")
                        break  #  Salir del bucle FOR si OK
                    else:
                        estado_salud = f"error_{response.status_code}"
                        logger.warning(
                            f"Validación de salud fallida para '{nombre}'. Status: {response.status_code}")

                # --- Bloque Except MUY Simplificado ---
                except Exception as e:
                    # Asignar un estado genérico y loguear la excepción
                    # completa
                    estado_salud = "error_inesperado"
                    logger.exception(
                        f"Error inesperado al validar salud de '{nombre}': {e}")
                    # --- FIN: Bloque if/elif/else indentado ---

                # Salir del bucle si el estado es 'ok' (ya estaba)
                if estado_salud == "ok":
                    break

                # --- Lógica de Reintento (YA ESTABA CORRECTAMENTE FUERA DEL TRY/EXCEPT) ---
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.debug(
                        f"Reintentando validación de salud para '{nombre}' en {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Validación de salud para '{nombre}' falló después de {MAX_RETRIES} intentos. Último estado: {estado_salud}")
            # --- FIN: Lógica de Validación (Fin del bucle for) -

        # --- Actualizar estado de salud (AHORA 'estado_salud' siempre tiene el valor correcto) ---
        with self.lock:
            if nombre in self.modules:
                self.modules[nombre]["estado_salud"] = estado_salud
            else:
                logger.warning(
                    f"Módulo '{nombre}' desapareció antes de actualizar estado de salud.")
                return

        # --- Notificación a Harmony Controller (AHORA usa el 'estado_salud' correcto) ---
        if (estado_salud == "ok" and
            modulo_tipo == "auxiliar" and
            modulo_aporta_a and
                modulo_naturaleza):
            if modulo_url_control:
                logger.info(
                    f"Módulo auxiliar '{nombre}' saludable. Notificando a Harmony Controller...")
                self._notify_harmony_controller_of_tool(
                    nombre=nombre,
                    url=modulo_url_control,
                    aporta_a=modulo_aporta_a,
                    naturaleza=modulo_naturaleza
                )
            else:
                logger.error(
                    f"Módulo auxiliar '{nombre}' saludable pero no tiene URL de control definida. No se puede notificar a HC.")
        elif estado_salud == "ok" and modulo_tipo == "auxiliar":
            if not modulo_aporta_a:
                logger.warning(
                    f"Módulo auxiliar '{nombre}' saludable pero no declaró 'aporta_a'. No se notificará a HC.")
            if not modulo_naturaleza:
                logger.warning(
                    f"Módulo auxiliar '{nombre}' saludable pero no declaró 'naturaleza_auxiliar'. No se notificará a HC.")

    # NUEVO: Función para notificar a HC sobre un nuevo tool auxiliar y
    # naturaleza ###
    def _notify_harmony_controller_of_tool(
            self,
            nombre: str,
            url: str,
            aporta_a: str,
            naturaleza: str):
        # Usar la URL de registro almacenada
        register_url = self.hc_register_url
        payload = {...}
        for attempt in range(MAX_RETRIES):
            try:
                logger.debug(
                    f"Notificando a HC sobre '{nombre}' (Naturaleza: {naturaleza}) en {register_url} (intento {attempt+1}/{MAX_RETRIES})")
                response = requests.post(
                    register_url, json=payload, timeout=REQUESTS_TIMEOUT)
                response.raise_for_status()
                logger.info(
                    f"Notificación para '{nombre}' enviada exitosamente a HC. Respuesta: {response.status_code}")
                return  #  Éxito -> Salir de la función
            # --- Bloque Except MUY Simplificado ---
            except Exception as e:
                logger.exception(f"Error inesperado: {e}")
                # --- Lógica de Espera/Reintento (DENTRO DEL BUCLE) ---
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_RETRY_DELAY * (2 ** attempt)
                    logger.debug(
                        f"Reintentando notificación a HC para '{nombre}' en {delay:.2f}s...")
                    time.sleep(delay)
                #  --- Fin Bloque Except ---
            else: # pylint: disable=useless-else-on-loop
                logger.error(
                    f"No se pudo notificar a HC sobre '{nombre}' después de {MAX_RETRIES} intentos.")

    # ... (actualizar_comando_estrategico, recibir_control_cogniboard, recibir_config_status sin cambios funcionales) ...
    def actualizar_comando_estrategico(self, comando, valor):
        """Procesa comandos de alto nivel que afectan la estrategia o configuración global."""
        with self.lock:
            if comando == "set_strategy":
                if isinstance(valor, str):
                    self.current_strategy = valor
                    logger.info(f"Estrategia actualizada a: {valor}")
                    return {"status": "success",
                            "mensaje": f"Estrategia establecida a '{valor}'"}
                else:
                    return {
                        "status": "error",
                        "mensaje": "Valor para 'set_strategy' debe ser un string"}
            elif comando == "set_target_setpoint_vector":
                try:
                    if valor is None:
                        return {
                            "status": "error",
                            "mensaje": "Falta 'valor' para 'set_target_setpoint_vector'"}
                    new_vector = [float(x) for x in valor]
                    self.target_setpoint_vector = new_vector
                    logger.info(
                        f"Setpoint objetivo directo actualizado a: {new_vector}")
                    self._send_setpoint_to_harmony(self.target_setpoint_vector)
                    return {
                        "status": "success",
                        "mensaje": f"Setpoint objetivo establecido a {new_vector}"}
                except (ValueError, TypeError):
                    return {
                        "status": "error",
                        "mensaje": "Valor para 'set_target_setpoint_vector' debe ser una lista de números"}
            else:
                logger.warning(
                    f"Comando estratégico desconocido recibido: {comando}")
                return {
                    "status": "error",
                    "mensaje": f"Comando estratégico '{comando}' no reconocido"}

    def recibir_control_cogniboard(self, control_signal):
        """Actualiza el estado interno con la señal de Cogniboard."""
        with self.lock:
            self.external_inputs["cogniboard_signal"] = control_signal
        logger.debug(
            f"Señal de control de Cogniboard actualizada: {control_signal}")

    def recibir_config_status(self, config_status):
        """Actualiza el estado interno con el status de Config Agent."""
        with self.lock:
            self.external_inputs["config_status"] = config_status
        logger.debug(f"Estado de configuración actualizado: {config_status}")

    # MODIFICADO: obtener_estado_completo ahora incluye detalles de módulos
    # ###

    def obtener_estado_completo(self):
        """Retorna una vista completa del estado interno de AgentAI."""
        with self.lock:
            modules_list = [dict(info) for info in self.modules.values()]
            harmony_state_copy = dict(self.harmony_state)
            external_inputs_copy = dict(self.external_inputs)
            target_setpoint_copy = list(self.target_setpoint_vector)
            current_strategy_copy = self.current_strategy

        return {
            "target_setpoint_vector": target_setpoint_copy,
            "current_strategy": current_strategy_copy,
            "external_inputs": external_inputs_copy,
            "harmony_controller_last_state": harmony_state_copy,
            # Incluye ahora 'naturaleza_auxiliar' si aplica
            "registered_modules": modules_list
        }

    def shutdown(self):
        """Detiene el bucle estratégico."""
        logger.info("Solicitando detención del bucle estratégico...")
        self._stop_event.set()
        if self._strategic_thread.is_alive():
            self._strategic_thread.join(timeout=STRATEGIC_LOOP_INTERVAL + 1)
            if self._strategic_thread.is_alive():
                logger.warning("El hilo estratégico no terminó limpiamente.")


# --- Instancia Global ---


# Crear la instancia de AgentAI ANTES de configurar las rutas que la usan
agent_ai_instance_app = AgentAI()  #  Crear instancia aquí
agent_api = Flask(__name__)

# Ahora las rutas pueden usar la instancia creada 'agent_ai_instance_app'


@agent_api.route('/api/register', methods=['POST'])
def handle_register():
    data = request.get_json()
    if not data:
        return jsonify(
            {"status": "error", "message": "No JSON data received"}), 400
    # Usar la instancia creada
    result = agent_ai_instance_app.registrar_modulo(data)
    status_code = 200 if result.get("status") == "success" else 400
    return jsonify(result), status_code


@agent_api.route('/api/state', methods=['GET'])
def handle_get_state():
    # Usar la instancia creada
    return jsonify(agent_ai_instance_app.obtener_estado_completo()), 200


@agent_api.route('/api/command', methods=['POST'])
def handle_command():
    data = request.get_json()
    if not data or 'comando' not in data or 'valor' not in data:
        return jsonify({"status": "error", "message": "Comando inválido"}), 400
    # Usar la instancia creada
    result = agent_ai_instance_app.actualizar_comando_estrategico(
        data['comando'], data['valor'])
    status_code = 200 if result.get("status") == "success" else 400
    return jsonify(result), status_code

# ... otros endpoints ...


def run_agent_ai_service():
    port = int(os.environ.get('AGENT_AI_PORT', 9000))
    logger.info(f"Iniciando servicio Flask para AgentAI en puerto {port}...")
    #  Iniciar el bucle estratégico de la instancia creada
    agent_ai_instance_app.start_loop()
    agent_api.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    try:
        run_agent_ai_service()
    except KeyboardInterrupt:
        logger.info("Interrupción manual.")
    finally:
        #  Apagar la instancia creada
        agent_ai_instance_app.shutdown()
        logger.info("AgentAI finalizado.")
