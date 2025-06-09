#!/usr/bin/env python3
"""
harmony_controller.py - Controlador Táctico del Ecosistema Watchers

- Implementa el bucle de control principal (basado en PID).
- Lee el estado de la matriz ECU.
- Recibe notificaciones de AgentAI para tools auxiliares gestionados
  (incluyendo naturaleza).  # MODIFICADO
- Recibe un setpoint (objetivo de armonía) de AgentAI.
- Calcula señales de control (aún no diferenciadas semánticamente)
  para cada watcher_tool gestionado.  # MODIFICADO
- Envía las señales de control a los watchers_tools.
- Expone su estado integrado y permite ajustar el setpoint vía API REST.
"""

import time
import threading
import logging
import requests
import numpy as np
import os
import json
from flask import Flask, jsonify, request
from typing import Dict, List, Any, Optional

from boson_phase import BosonPhase

ECU_API_URL = os.environ.get("ECU_API_URL", "http://ecu:8000/api/ecu")

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s [%(levelname)s] [%(threadName)s] "
        "%(name)s: %(message)s"
    ),
)
logger = logging.getLogger("harmony_controller")

# --- Configuración del Controlador ---

# --- MODIFICADO: MANAGED_TOOLS ya no se lee de ENV, se poblará dinámicamente ---
# WATCHERS_TOOLS_JSON = os.environ.get('WATCHERS_TOOLS_CONFIG', '{}')
# try:
#     MANAGED_TOOLS_INIT: Dict[str, str] = json.loads(WATCHERS_TOOLS_JSON)
#     logger.info(
#         f"Watchers Tools iniciales (config): {list(MANAGED_TOOLS_INIT.keys())}"
#     )
# except json.JSONDecodeError:
#     logger.warning(
#         "No se proporcionó WATCHERS_TOOLS_CONFIG o es inválido. "
#         "HC esperará registros de AgentAI."
#     )
#     MANAGED_TOOLS_INIT = {}
# Ahora HC dependerá de AgentAI para saber qué tools gestionar.

KP_INIT = float(os.environ.get("HC_KP", 1.0))
KI_INIT = float(os.environ.get("HC_KI", 0.1))
KD_INIT = float(os.environ.get("HC_KD", 0.05))

SETPOINT_VECTOR_JSON = os.environ.get(
    "HC_SETPOINT_VECTOR", "[1.0, 0.0]"
)  # Ajustado default
try:
    setpoint_vector_init = np.array(
        json.loads(SETPOINT_VECTOR_JSON), dtype=float
    )
    setpoint_init = np.linalg.norm(setpoint_vector_init)
    logger.info(
        "Setpoint inicial (norma): %.3f (derivado de %s)",
        setpoint_init, setpoint_vector_init.tolist()
    )
except (json.JSONDecodeError, ValueError):
    logger.error(
        "Error al procesar HC_SETPOINT_VECTOR: %s. Usando setpoint=1.0",
        SETPOINT_VECTOR_JSON
    )
    setpoint_vector_init = np.array(
        [1.0, 0.0]
    )  # Default consistente con AgentAI
    setpoint_init = 1.0

CONTROL_LOOP_INTERVAL = float(os.environ.get("HC_INTERVAL", 1.0))
REQUESTS_TIMEOUT = float(os.environ.get("HC_REQUESTS_TIMEOUT", 2.0))
MAX_RETRIES = int(os.environ.get("HC_MAX_RETRIES", 3))
BASE_RETRY_DELAY = float(os.environ.get("HC_BASE_RETRY_DELAY", 0.5))


# --- Estado Global del Controlador ---
class HarmonyControllerState:
    def __init__(
        self,
        kp=KP_INIT,
        ki=KI_INIT,
        kd=KD_INIT,
        initial_setpoint=setpoint_init,
        initial_setpoint_vector=setpoint_vector_init,
    ):
        self.pid_controller = BosonPhase(kp, ki, kd, setpoint=initial_setpoint)
        self.current_setpoint = initial_setpoint
        self.setpoint_vector = (
            initial_setpoint_vector.tolist()
            if isinstance(initial_setpoint_vector, np.ndarray)
            else initial_setpoint_vector
        )
        self.last_ecu_state: List[List[float]] = []
        # --- MODIFICADO: Estructura para almacenar detalles, incluyendo naturaleza ---
        # Clave: nombre_tool, Valor: Dict{'url': str, 'aporta_a': str, 'naturaleza': str,
        # 'last_state': Any, 'last_control': float}
        self.managed_tools_details: Dict[str, Dict[str, Any]] = {}
        self.last_measurement: float = 0.0
        self.last_pid_output: float = 0.0
        self.lock = threading.Lock()

    # ... (update_setpoint sin cambios) ...
    def update_setpoint(
        self,
        new_setpoint_value: float,
        new_setpoint_vector: Optional[List[float]] = None,
    ):
        with self.lock:
            self.current_setpoint = new_setpoint_value
            self.pid_controller.setpoint = new_setpoint_value
            if new_setpoint_vector:
                self.setpoint_vector = new_setpoint_vector
            logger.info(
                f"Setpoint actualizado a: {self.current_setpoint:.3f}"
                + (
                    f" desde vector {new_setpoint_vector}"
                    if new_setpoint_vector
                    else ""
                )
            )

    # MODIFICADO: register_managed_tool ahora acepta y almacena 'naturaleza'
    # ###
    def register_managed_tool(
        self, nombre: str, url: str, aporta_a: str, naturaleza: str
    ):  # MODIFICADO
        with self.lock:
            if nombre not in self.managed_tools_details:
                self.managed_tools_details[nombre] = {}
                logger.info(
                    "Registrando tool: '%s' (URL: %s, Aporta: %s, Nat: %s)",
                    nombre, url, aporta_a, naturaleza
                )  # MODIFICADO
            else:
                logger.info(
                    "Actualizando información del tool gestionado: '%s'", nombre
                )
            self.managed_tools_details[nombre]["url"] = url
            self.managed_tools_details[nombre]["aporta_a"] = aporta_a
            self.managed_tools_details[nombre][
                "naturaleza"
            ] = naturaleza  # NUEVO ###
            self.managed_tools_details[nombre]["last_state"] = {
                "status": "unknown"
            }
            self.managed_tools_details[nombre]["last_control"] = 0.0

    # ... (unregister_managed_tool sin cambios) ...
    def unregister_managed_tool(self, nombre: str):
        with self.lock:
            if nombre in self.managed_tools_details:
                logger.info(f"Eliminando tool gestionado: '{nombre}'")
                del self.managed_tools_details[nombre]
            else:
                logger.warning(
                    f"Intento de eliminar tool no gestionado: '{nombre}'"
                )

    ### MODIFICADO: get_state_snapshot ahora incluye 'naturaleza' ###
    def get_state_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            tools_snapshot = {}
            for name, details in self.managed_tools_details.items():
                tools_snapshot[name] = {
                    "url": details.get("url"),
                    "aporta_a": details.get("aporta_a"),
                    "naturaleza": details.get("naturaleza"),  # NUEVO ###
                    "last_state": details.get(
                        "last_state", {"status": "unknown"}
                    ),
                    "last_control": details.get("last_control", 0.0),
                }

            return {
                "setpoint_value": self.current_setpoint,
                "setpoint_vector": list(self.setpoint_vector),
                "last_measurement": self.last_measurement,
                "last_pid_output": self.last_pid_output,
                "last_ecu_state": self.last_ecu_state,
                "managed_tools": tools_snapshot,  # Incluye naturaleza
                "pid_gains": {
                    "Kp": self.pid_controller.Kp,
                    "Ki": self.pid_controller.Ki,
                    "Kd": self.pid_controller.Kd,
                },
            }


# Instancia global del estado
controller_state = HarmonyControllerState()


# --- Funciones Auxiliares de Comunicación (get_ecu_state, get_tool_state, send_tool_control sin cambios funcionales) ---
def get_ecu_state() -> Optional[List[List[float]]]:
    """Obtiene el estado unificado de la ECU vía API REST con reintentos."""
    url = ECU_API_URL
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                state_list = data.get("estado_campo_unificado")
                if isinstance(state_list, list) and all(
                    isinstance(row, list) for row in state_list
                ):
                    logger.debug(
                        f"Estado ECU recibido: {len(state_list)}x{len(state_list[0]) if state_list else 0} puntos"
                    )
                    return state_list
                else:
                    logger.error(
                        f"Clave 'estado_campo_unificado' encontrada pero no es lista de listas: {type(state_list)}"
                    )
            else:
                logger.warning(
                    f"Respuesta de ECU no exitosa: {data.get('message', 'Formato desconocido')}"
                )
        except BaseException as e:
            logger.error(
                f"Error al obtener/procesar estado de ECU ({url}) intento {attempt+1}: {type(e).__name__} - {e}"
            )
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (2**attempt)
                time.sleep(delay)
    return None


def get_tool_state(tool_name: str, base_url: str) -> Dict[str, Any]:
    """Obtiene el estado de un watcher_tool específico vía API REST con reintentos."""
    state_url = f"{base_url}/api/state"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(state_url, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            logger.debug(
                f"Estado recibido de {tool_name}: {data.get('state', data)}"
            )
            return data.get("state", {"status": "success", "raw_data": data})
        except Exception as e:
            logger.warning(
                f"Error al obtener estado de {tool_name} ({state_url}) intento {attempt+1}: {type(e).__name__} - {e}"
            )
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (2**attempt)
                time.sleep(delay)
    return {
        "status": "error",
        "message": f"No se pudo obtener estado después de {MAX_RETRIES} intentos",
    }


def send_tool_control(tool_name: str, base_url: str, control_signal: float):
    """Envía una señal de control a un watcher_tool específico vía API REST con reintentos."""
    control_url = f"{base_url}/api/control"
    payload = {"control_signal": control_signal}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                control_url, json=payload, timeout=REQUESTS_TIMEOUT
            )
            response.raise_for_status()
            logger.info(
                f"Señal de control {control_signal:.3f} enviada a {tool_name}. Respuesta: {response.status_code}"
            )
            return True
        except Exception as e:
            logger.warning(
                f"Error al enviar control a {tool_name} ({control_url}) intento {attempt+1}: {type(e).__name__} - {e}"
            )
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (2**attempt)
                time.sleep(delay)
    logger.error(
        f"No se pudo enviar señal de control a {tool_name} después de {MAX_RETRIES} intentos."
    )
    return False


# --- Bucle de Control Principal (Hilo) ---
def harmony_control_loop():
    """Hilo de fondo que ejecuta el bucle de control táctico."""
    logger.info("Iniciando bucle de control Harmony...")

    affinity_to_setpoint_index = {
        "malla_watcher": 0,
        "matriz_ecu": 1,
    }
    num_control_axes = len(affinity_to_setpoint_index)

    while True:
        start_time = time.monotonic()
        dt = CONTROL_LOOP_INTERVAL

        with controller_state.lock:
            current_sp_norm = controller_state.current_setpoint
            current_setpoint_vector = list(controller_state.setpoint_vector)
            managed_tools_loop_copy = dict(
                controller_state.managed_tools_details
            )

        # 1. Leer Estado ECU
        ecu_state_data = get_ecu_state()

        # 2. Calcular Medición Agregada (desde ECU)
        # ... (lógica sin cambios) ...
        current_measurement = 0.0
        last_ecu_state_to_store = []
        if ecu_state_data is not None:
            try:
                ecu_state_array = np.array(ecu_state_data, dtype=float)
                current_measurement = (
                    np.linalg.norm(ecu_state_array)
                    if ecu_state_array.size > 0
                    else 0.0
                )
                last_ecu_state_to_store = ecu_state_data
                logger.debug(
                    f"[ControlLoop] ECU State Norm={current_measurement:.3f}"
                )
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Error al procesar datos de ECU: {e} - Data: {ecu_state_data}"
                )
                current_measurement = 0.0
                last_ecu_state_to_store = []
        else:
            logger.warning(
                "[ControlLoop] No se pudo obtener estado válido de ECU, usando medición 0.0"
            )
            current_measurement = 0.0
            last_ecu_state_to_store = []

        # 3. Calcular Salida PID Global
        pid_output = 0.0
        if hasattr(controller_state, "pid_controller"):
            pid_output = controller_state.pid_controller.compute(
                current_sp_norm, current_measurement, dt
            )
            logger.debug(
                f"[ControlLoop] SetpointNorm={current_sp_norm:.3f}, Measurement={current_measurement:.3f}, PID Output={pid_output:.3f}"
            )
        else:
            logger.error("Instancia PID no encontrada en controller_state.")

        # 4. Calcular Pesos y Enviar Señales de Control Específicas
        control_weights = [
            1.0 / num_control_axes
        ] * num_control_axes  # Default: pesos iguales
        if len(current_setpoint_vector) >= num_control_axes:
            relevant_setpoint_components = [
                current_setpoint_vector[i] for i in range(num_control_axes)
            ]
            abs_components = np.abs(relevant_setpoint_components)
            sum_abs_components = np.sum(abs_components)
            if sum_abs_components > 1e-9:
                control_weights = abs_components / sum_abs_components
                logger.debug(
                    f"[ControlLoop] SetpointVector (relevant): {relevant_setpoint_components}, Calculated Weights: {control_weights.tolist()}"
                )
            else:
                logger.debug(
                    f"[ControlLoop] SetpointVector componentes relevantes cero. Usando pesos iguales."
                )
                # Mantener pesos iguales ya asignados
        else:
            logger.warning(
                f"[ControlLoop] SetpointVector (len={len(current_setpoint_vector)}) "
                f"no tiene suficientes dimensiones ({num_control_axes}). Usando pesos iguales.")
            # Mantener pesos iguales ya asignados

        current_tools_state: Dict[str, Any] = {}
        control_signals_sent: Dict[str, float] = {}

        for name, details in managed_tools_loop_copy.items():
            tool_url = details.get("url")
            tool_aporta_a = details.get("aporta_a")
            tool_naturaleza = details.get(
                "naturaleza"
            )  # NUEVO: Obtener naturaleza ###

            if not tool_url:
                logger.error(f"No hay URL definida para '{name}'. Saltando.")
                current_tools_state[name] = {
                    "status": "error",
                    "message": "URL no definida",
                }
                continue

            # --- Calcular Señal Específica (CON LÓGICA DE NATURALEZA) ---
        signal_to_send = 0.0  # Default seguro
        control_index = affinity_to_setpoint_index.get(tool_aporta_a)

        if control_index is not None:
            weight = control_weights[control_index]
            base_signal = pid_output * weight
            # --- INICIO: Lógica basada en tool_naturaleza ---
            if tool_naturaleza == "potenciador":
                # Si PID pide aumentar (>0), potenciar más (señal base).
                # Si PID pide reducir (<0), potenciar menos (señal base).
                signal_to_send = base_signal
                log_nature_logic = f"signal = base ({base_signal:.3f})"
            elif tool_naturaleza == "reductor":
                # Si PID pide reducir (<0), enviar señal positiva (reducir más).
                # Si PID pide aumentar (>0), enviar señal negativa (reducir
                # menos).
                signal_to_send = -base_signal  # Invertir señal base
                log_nature_logic = f"signal = -base ({-base_signal:.3f})"
            elif (
                tool_naturaleza == "modulador" or tool_naturaleza == "actuador"
            ):
                # Pasar señal base, el tool la interpreta.
                signal_to_send = base_signal
                log_nature_logic = f"signal = base ({base_signal:.3f})"
            elif (
                tool_naturaleza == "sensor" or tool_naturaleza == "convertidor"
            ):
                # No controlar activamente.
                signal_to_send = 0.0
                log_nature_logic = "signal = 0.0 (pasivo)"
            else:
                # Naturaleza desconocida.
                logger.warning(
                    f"[CtrlDiff] Naturaleza '{tool_naturaleza}' desconocida para '{name}'. Enviando señal 0.0."
                )
                signal_to_send = 0.0
                log_nature_logic = "signal = 0.0 (desconocido)"
                # --- FIN: Lógica basada en tool_naturaleza ---

                logger.debug(
                    f"[CtrlDiff] Tool '{name}' (Aporta: {tool_aporta_a}, Nat: {tool_naturaleza}, Idx: {control_index}): "
                    f"Weight={weight:.3f}, PIDOut={pid_output:.3f}, BaseSig={base_signal:.3f} -> Logic: {log_nature_logic} -> FinalSignal={signal_to_send:.3f}")
        else:  # control_index is None
            logger.warning(
                f"[CtrlDiff] Afinidad '{tool_aporta_a}' para tool '{name}' no mapeada. Enviando señal 0.0."
            )
            signal_to_send = 0.0
            # --- Fin Calcular Señal Específica ---

            if send_tool_control(name, tool_url, signal_to_send):
                control_signals_sent[name] = signal_to_send
            else:
                control_signals_sent[name] = None  # Indicar fallo

        # 5. Actualizar Estado Global (protegido por lock)
        with controller_state.lock:
            controller_state.last_ecu_state = last_ecu_state_to_store
            controller_state.last_measurement = current_measurement
            controller_state.last_pid_output = pid_output
            # Actualizar estado y control para cada tool en la estructura
            # principal
            for name, state_data in current_tools_state.items():
                if name in controller_state.managed_tools_details:
                    controller_state.managed_tools_details[name][
                        "last_state"
                    ] = state_data
            for name, control_val in control_signals_sent.items():
                if name in controller_state.managed_tools_details:
                    # Guardar el valor enviado, o un marcador si falló
                    controller_state.managed_tools_details[name][
                        "last_control"
                    ] = (
                        control_val
                        if control_val is not None
                        else "send_failed"
                    )

        # Esperar
        elapsed_time = time.monotonic() - start_time
        sleep_time = max(0, dt - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            logger.warning(
                f"El ciclo de control ({elapsed_time:.3f}s) excedió el intervalo ({dt}s)."
            )


# --- API Flask ---
app = Flask(__name__)


# ... (Endpoints /api/health, /api/harmony/state, /api/harmony/setpoint sin cambios) ...
@app.route("/api/health", methods=["GET"])
def health_check():
    """Endpoint para verificar la salud del servicio."""
    control_loop_alive = any(
        t.name == "HarmonyControlLoop" and t.is_alive()
        for t in threading.enumerate()
    )
    return (
        jsonify(
            {
                "status": "healthy" if control_loop_alive else "warning",
                "service": "harmony_controller",
                "control_loop_running": control_loop_alive,
                "timestamp": time.time(),
            }
        ),
        200,
    )


@app.route("/api/harmony/state", methods=["GET"])
def get_harmony_state():
    """Devuelve el estado actual del controlador Harmony."""
    return (
        jsonify(
            {
                "status": "success",
                "data": controller_state.get_state_snapshot(),
            }
        ),
        200,
    )


@app.route("/api/harmony/setpoint", methods=["POST"])
def set_harmony_setpoint():
    """Permite actualizar el setpoint del controlador desde AgentAI."""
    data = request.get_json()
    if not data:
        return (
            jsonify(
                {"status": "error", "message": "Payload JSON vacío o ausente."}
            ),
            400,
        )
    new_value = data.get("setpoint_value")
    new_vector = data.get("setpoint_vector")
    try:
        if new_vector is not None and isinstance(new_vector, list):
            vec = np.array(new_vector, dtype=float)
            val = np.linalg.norm(vec)
            controller_state.update_setpoint(val, vec.tolist())
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": f"Setpoint actualizado a norma({val:.3f}) desde vector.",
                        "new_setpoint_value": val,
                        "new_setpoint_vector": vec.tolist(),
                    }),
                200,
            )
        elif new_value is not None:
            val = float(new_value)
            controller_state.update_setpoint(val, None)
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": "Setpoint actualizado a valor.",
                        "new_setpoint_value": val,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Se requiere 'setpoint_value' o 'setpoint_vector' en el JSON.",
                    }),
                400,
            )
    except (ValueError, TypeError) as e:
        logger.error(f"Error al procesar nuevo setpoint: {e}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Error en el formato de los datos: {e}",
                }
            ),
            400,
        )
    except Exception:  # F841: e is not used when logger.exception is called
        logger.exception("Error inesperado al actualizar setpoint.")
        return (
            jsonify(
                {"status": "error", "message": "Error interno del servidor."}
            ),
            500,
        )


# MODIFICADO: Endpoint /api/harmony/register_tool ahora espera
# 'naturaleza' ###
@app.route("/api/harmony/register_tool", methods=["POST"])
def register_tool_from_ai():
    """
    Endpoint para que AgentAI notifique sobre un tool auxiliar gestionable.
    Espera JSON: {"nombre": str, "url": str, "aporta_a": str, "naturaleza": str}
    # MODIFICADO
    """
    data = request.get_json()
    if not data:
        logger.error("Solicitud a /register_tool sin payload JSON.")
        return (
            jsonify(
                {"status": "error", "message": "Payload JSON vacío o ausente."}
            ),
            400,
        )

    nombre = data.get("nombre")
    url = data.get("url")
    aporta_a = data.get("aporta_a")
    naturaleza = data.get("naturaleza")  # NUEVO ###

    # Validación básica (incluyendo naturaleza)
    if not nombre or not isinstance(nombre, str):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Campo 'nombre' ausente o inválido.",
                }
            ),
            400,
        )
    if not url or not isinstance(url, str):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Campo 'url' ausente o inválido.",
                }
            ),
            400,
        )
    if not aporta_a or not isinstance(aporta_a, str):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Campo 'aporta_a' ausente o inválido.",
                }
            ),
            400,
        )
    ### NUEVO: Validar naturaleza ###
    if not naturaleza or not isinstance(naturaleza, str):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Campo 'naturaleza' ausente o inválido.",
                }
            ),
            400,
        )
    # Podrías añadir validación contra una lista de naturalezas permitidas si quieres ser estricto
    # naturalezas_validas = ["potenciador", "reductor", "modulador", "sensor", "actuador", "convertidor"]
    # if naturaleza not in naturalezas_validas:
    # return jsonify({"status": "error", "message": f"Naturaleza
    # '{naturaleza}' no reconocida."}), 400

    try:
        # Pasar naturaleza al método de registro
        controller_state.register_managed_tool(
            nombre, url, aporta_a, naturaleza
        )  # MODIFICADO ###
        return (
            jsonify(
                {
                    "status": "success",
                    "message": f"Tool '{nombre}' registrado/actualizado en Harmony Controller.",
                }),
            200,
        )
    except Exception:  # F841: e is not used when logger.exception is called
        logger.exception(f"Error al registrar tool '{nombre}' desde AgentAI.")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Error interno al registrar el tool.",
                }
            ),
            500,
        )


# ... (Endpoint /api/harmony/pid/reset sin cambios) ...
@app.route("/api/harmony/pid/reset", methods=["POST"])
def reset_pid():
    """Resetea los acumuladores del controlador PID."""
    try:
        if hasattr(controller_state, "pid_controller"):
            controller_state.pid_controller.reset()
            return (
                jsonify({"status": "success", "message": "PID reiniciado."}),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Controlador PID no encontrado.",
                    }
                ),
                500,
            )
    except Exception:  # F841: e is not used when logger.exception is called
        logger.exception("Error inesperado al reiniciar PID.")
        return (
            jsonify(
                {"status": "error", "message": "Error interno del servidor."}
            ),
            500,
        )


# --- Punto de Entrada (sin cambios) ---
def main():
    control_thread = threading.Thread(
        target=harmony_control_loop, daemon=True, name="HarmonyControlLoop"
    )
    control_thread.start()
    port = int(os.environ.get("HC_PORT", 7000))
    logger.info(
        f"Iniciando servidor Flask para Harmony Controller en puerto {port}..."
    )
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
