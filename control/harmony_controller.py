#!/usr/bin/env python3
"""
harmony_controller.py - Controlador Táctico del Ecosistema Watchers

- Implementa el bucle de control principal (basado en PID).
- Lee el estado de la matriz ECU.
- Recibe notificaciones de AgentAI para tools auxiliares gestionados
  (incluyendo naturaleza).
- Recibe un setpoint (objetivo de armonía) de AgentAI.
- Calcula señales de control para cada watcher_tool gestionado.
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
KP_INIT = float(os.environ.get("HC_KP", 1.0))
KI_INIT = float(os.environ.get("HC_KI", 0.1))
KD_INIT = float(os.environ.get("HC_KD", 0.05))

SETPOINT_VECTOR_JSON = os.environ.get("HC_SETPOINT_VECTOR", "[1.0, 0.0]")
try:
    setpoint_vector_init = np.array(
        json.loads(SETPOINT_VECTOR_JSON), dtype=float
    )
    setpoint_init = np.linalg.norm(setpoint_vector_init)
    logger.info(
        "SP inicial (norma): %.3f (vec: %s)",
        setpoint_init, setpoint_vector_init.tolist()
    )
except (json.JSONDecodeError, ValueError):
    logger.error(
        "Error HC_SETPOINT_VECTOR: %s. Usando SP=1.0", SETPOINT_VECTOR_JSON
    )
    setpoint_vector_init = np.array([1.0, 0.0])
    setpoint_init = 1.0

CONTROL_LOOP_INTERVAL = float(os.environ.get("HC_INTERVAL", 1.0))
REQUESTS_TIMEOUT = float(os.environ.get("HC_REQUESTS_TIMEOUT", 2.0))
MAX_RETRIES = int(os.environ.get("HC_MAX_RETRIES", 3))
BASE_RETRY_DELAY = float(os.environ.get("HC_BASE_RETRY_DELAY", 0.5))


class HarmonyControllerState:
    """Almacena el estado global del controlador Harmony."""

    def __init__(
        self,
        kp=KP_INIT,
        ki=KI_INIT,
        kd=KD_INIT,
        initial_setpoint=setpoint_init,
        initial_setpoint_vector=setpoint_vector_init,
    ):
        self.pid_controller = BosonPhase(
            kp, ki, kd, setpoint=initial_setpoint
        )
        self.current_setpoint = initial_setpoint
        if isinstance(initial_setpoint_vector, np.ndarray):
            self.setpoint_vector = initial_setpoint_vector.tolist()
        else:
            self.setpoint_vector = initial_setpoint_vector
        self.last_ecu_state: List[List[float]] = []
        self.managed_tools_details: Dict[str, Dict[str, Any]] = {}
        self.last_measurement: float = 0.0
        self.last_pid_output: float = 0.0
        self.lock = threading.Lock()

    def update_setpoint(
        self,
        new_setpoint_value: float,
        new_setpoint_vector: Optional[List[float]] = None,
    ):
        """Actualiza el setpoint del controlador PID."""
        with self.lock:
            self.current_setpoint = new_setpoint_value
            self.pid_controller.setpoint = new_setpoint_value
            if new_setpoint_vector:
                self.setpoint_vector = new_setpoint_vector
            log_msg = f"Setpoint actualizado a: {self.current_setpoint:.3f}"
            if new_setpoint_vector:
                log_msg += f" desde vector {new_setpoint_vector}"
            logger.info(log_msg)

    def register_managed_tool(
        self, nombre: str, url: str, aporta_a: str, naturaleza: str
    ):
        """Registra o actualiza un watcher tool gestionado."""
        with self.lock:
            if nombre not in self.managed_tools_details:
                self.managed_tools_details[nombre] = {}
                logger.info(
                    "Registrando tool: '%s' (URL: %s, Aporta: %s, Nat: %s)",
                    nombre, url, aporta_a, naturaleza
                )
            else:
                logger.info(
                    "Actualizando información del tool: '%s'", nombre
                )
            self.managed_tools_details[nombre]["url"] = url
            self.managed_tools_details[nombre]["aporta_a"] = aporta_a
            self.managed_tools_details[nombre]["naturaleza"] = naturaleza
            self.managed_tools_details[nombre]["last_state"] = {
                "status": "unknown"
            }
            self.managed_tools_details[nombre]["last_control"] = 0.0

    def unregister_managed_tool(self, nombre: str):
        """Elimina un watcher tool de la gestión."""
        with self.lock:
            if nombre in self.managed_tools_details:
                logger.info("Eliminando tool gestionado: '%s'", nombre)
                del self.managed_tools_details[nombre]
            else:
                logger.warning(
                    "Intento de eliminar tool no gestionado: '%s'", nombre
                )

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Obtiene una instantánea del estado actual del controlador."""
        with self.lock:
            tools_snapshot = {
                name: {
                    "url": details.get("url"),
                    "aporta_a": details.get("aporta_a"),
                    "naturaleza": details.get("naturaleza"),
                    "last_state": details.get(
                        "last_state", {"status": "unknown"}
                    ),
                    "last_control": details.get("last_control", 0.0),
                }
                for name, details in self.managed_tools_details.items()
            }

            return {
                "setpoint_value": self.current_setpoint,
                "setpoint_vector": list(self.setpoint_vector),
                "last_measurement": self.last_measurement,
                "last_pid_output": self.last_pid_output,
                "last_ecu_state": self.last_ecu_state,
                "managed_tools": tools_snapshot,
                "pid_gains": {
                    "Kp": self.pid_controller.Kp,
                    "Ki": self.pid_controller.Ki,
                    "Kd": self.pid_controller.Kd,
                },
            }


controller_state = HarmonyControllerState()


def get_ecu_state() -> Optional[List[List[float]]]:
    """Obtiene el estado unificado de la ECU vía API REST."""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(ECU_API_URL, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success":
                state_list = data.get("estado_campo_unificado")
                if isinstance(state_list, list) and all(
                    isinstance(row, list) for row in state_list
                ):
                    rows = len(state_list)
                    cols = len(state_list[0]) if rows > 0 else 0
                    logger.debug("Estado ECU recibido: %dx%d puntos", rows, cols)
                    return state_list
                else:
                    logger.error(
                        "Clave 'estado_campo_unificado' no es lista de listas: %s",
                        type(state_list)
                    )
            else:
                msg = data.get('message', 'Formato desconocido')
                logger.warning("Respuesta de ECU no exitosa: %s", msg)
        except BaseException as e:
            logger.error(
                "Error al obtener estado de ECU (%s) intento %d: %s",
                ECU_API_URL, attempt + 1, e
            )
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (2**attempt)
                time.sleep(delay)
    return None


def get_tool_state(tool_name: str, base_url: str) -> Dict[str, Any]:
    """Obtiene el estado de un watcher_tool específico."""
    state_url = f"{base_url}/api/state"
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(state_url, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            logger.debug("Estado recibido de %s: %s",
                         tool_name, data.get('state', data))
            return data.get("state", {"status": "success", "raw_data": data})
        except Exception as e:
            logger.warning(
                "Error al obtener estado de %s (%s) intento %d: %s",
                tool_name, state_url, attempt + 1, e
            )
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (2**attempt)
                time.sleep(delay)
    return {
        "status": "error",
        "message": f"No se pudo obtener estado tras {MAX_RETRIES} intentos",
    }


def send_tool_control(tool_name: str, base_url: str, control_signal: float):
    """Envía señal de control a un watcher_tool específico."""
    control_url = f"{base_url}/api/control"
    payload = {"control_signal": control_signal}
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                control_url, json=payload, timeout=REQUESTS_TIMEOUT
            )
            response.raise_for_status()
            logger.info(
                "Señal de control %.3f enviada a %s. Respuesta: %d",
                control_signal, tool_name, response.status_code
            )
            return True
        except Exception as e:
            logger.warning(
                "Error al enviar control a %s (%s) intento %d: %s",
                tool_name, control_url, attempt + 1, e
            )
            if attempt < MAX_RETRIES - 1:
                delay = BASE_RETRY_DELAY * (2**attempt)
                time.sleep(delay)
    logger.error(
        "No se pudo enviar señal de control a %s tras %d intentos.",
        tool_name, MAX_RETRIES
    )
    return False


def harmony_control_loop():
    """Hilo de fondo que ejecuta el bucle de control táctico."""
    logger.info("Iniciando bucle de control Harmony...")

    affinity_to_setpoint_index = {"malla_watcher": 0, "matriz_ecu": 1}
    num_control_axes = len(affinity_to_setpoint_index)

    while True:
        start_time = time.monotonic()
        dt = CONTROL_LOOP_INTERVAL

        with controller_state.lock:
            current_sp_norm = controller_state.current_setpoint
            current_sp_vec = list(controller_state.setpoint_vector)
            tools_copy = dict(controller_state.managed_tools_details)

        ecu_state_data = get_ecu_state()

        current_measurement = 0.0
        last_ecu_state_to_store = []
        if ecu_state_data is not None:
            try:
                ecu_array = np.array(ecu_state_data, dtype=float)
                current_measurement = (
                    np.linalg.norm(ecu_array) if ecu_array.size > 0 else 0.0
                )
                last_ecu_state_to_store = ecu_state_data
                logger.debug(
                    "[ControlLoop] ECU State Norm=%.3f", current_measurement
                )
            except (ValueError, TypeError) as e:
                logger.error("Error al procesar datos de ECU: %s", e)
        else:
            logger.warning(
                "[ControlLoop] No se pudo obtener estado de ECU, usando 0.0"
            )

        pid_output = 0.0
        if hasattr(controller_state, "pid_controller"):
            pid_output = controller_state.pid_controller.compute(
                current_sp_norm, current_measurement, dt
            )
            logger.debug(
                "[CtrlLoop] SP=%.3f, PV=%.3f, PIDOut=%.3f",
                current_sp_norm, current_measurement, pid_output
            )
        else:
            logger.error("Instancia PID no encontrada en controller_state.")

        control_weights = [1.0 / num_control_axes] * num_control_axes
        if len(current_sp_vec) >= num_control_axes:
            rel_sp_comps = [current_sp_vec[i] for i in range(num_control_axes)]
            abs_comps = np.abs(rel_sp_comps)
            sum_abs_comps = np.sum(abs_comps)
            if sum_abs_comps > 1e-9:
                control_weights = abs_comps / sum_abs_comps
                logger.debug(
                    "[CtrlLoop] RelSPComps: %s, Wgts: %s",
                    rel_sp_comps, control_weights.tolist()
                )
            else:
                logger.debug("[CtrlLoop] RelSPComps zero. Equal wgts.")
        else:
            logger.warning(
                "[CtrlLoop] SPVec (len=%d) insuff. dims (%d). Eq Wgts.",
                len(current_sp_vec), num_control_axes
            )

        control_signals_sent: Dict[str, Any] = {}

        for name, details in tools_copy.items():
            tool_url = details.get("url")
            aporta_a = details.get("aporta_a")
            naturaleza = details.get("naturaleza")

            if not tool_url:
                logger.error("No hay URL definida para '%s'. Saltando.", name)
                continue

            signal_to_send = 0.0
            control_index = affinity_to_setpoint_index.get(aporta_a)

            if control_index is not None:
                weight = control_weights[control_index]
                base_signal = pid_output * weight
                if naturaleza == "reductor":
                    signal_to_send = -base_signal
                elif naturaleza in ("sensor", "convertidor"):
                    signal_to_send = 0.0
                else:
                    signal_to_send = base_signal
            else:
                logger.warning(
                    "[CtrlDiff] Afinidad '%s'/'%s' sin mapeo. Sig 0.0.",
                    aporta_a, name
                )

            if send_tool_control(name, tool_url, signal_to_send):
                control_signals_sent[name] = signal_to_send
            else:
                control_signals_sent[name] = "send_failed"

        with controller_state.lock:
            controller_state.last_ecu_state = last_ecu_state_to_store
            controller_state.last_measurement = current_measurement
            controller_state.last_pid_output = pid_output
            for name, control_val in control_signals_sent.items():
                if name in controller_state.managed_tools_details:
                    controller_state.managed_tools_details[name][
                        "last_control"
                    ] = control_val

        elapsed_time = time.monotonic() - start_time
        sleep_time = max(0, dt - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            logger.warning(
                "El ciclo de control (%.3f)s excedió el intervalo (%.3f)s.",
                elapsed_time, dt
            )


# --- API Flask ---
app = Flask(__name__)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Endpoint para verificar la salud del servicio."""
    control_loop_alive = any(
        t.name == "HarmonyControlLoop" and t.is_alive()
        for t in threading.enumerate()
    )
    return jsonify({
        "status": "healthy" if control_loop_alive else "warning",
        "service": "harmony_controller",
        "control_loop_running": control_loop_alive,
        "timestamp": time.time(),
    }), 200


@app.route("/api/harmony/state", methods=["GET"])
def get_harmony_state():
    """Devuelve el estado actual del controlador Harmony."""
    return jsonify({
        "status": "success",
        "data": controller_state.get_state_snapshot()
    }), 200


@app.route("/api/harmony/setpoint", methods=["POST"])
def set_harmony_setpoint():
    """Permite actualizar el setpoint del controlador desde AgentAI."""
    data = request.get_json()
    if not data:
        return jsonify({
            "status": "error", "message": "Payload JSON vacío o ausente."
        }), 400
    new_vector = data.get("setpoint_vector")
    try:
        if new_vector is not None and isinstance(new_vector, list):
            vec = np.array(new_vector, dtype=float)
            val = np.linalg.norm(vec)
            controller_state.update_setpoint(val, vec.tolist())
            return jsonify({
                "status": "success",
                "message": f"Setpoint norma({val:.3f}) desde vector.",
                "new_setpoint_value": val,
                "new_setpoint_vector": vec.tolist(),
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Se requiere 'setpoint_vector' en el JSON."
            }), 400
    except (ValueError, TypeError) as e:
        logger.error("Error al procesar nuevo setpoint: %s", e)
        return jsonify({
            "status": "error",
            "message": f"Error en el formato de los datos: {e}",
        }), 400


@app.route("/api/harmony/register_tool", methods=["POST"])
def register_tool_from_ai():
    """Endpoint para que AgentAI notifique sobre un tool gestionable."""
    data = request.get_json()
    if not data:
        logger.error("Solicitud a /register_tool sin payload JSON.")
        return jsonify({
            "status": "error", "message": "Payload JSON vacío o ausente."
        }), 400

    required_fields = ["nombre", "url", "aporta_a", "naturaleza"]
    if not all(field in data for field in required_fields):
        missing = [f for f in required_fields if f not in data]
        return jsonify({
            "status": "error",
            "message": f"Faltan campos requeridos: {', '.join(missing)}"
        }), 400

    try:
        controller_state.register_managed_tool(
            data["nombre"], data["url"], data["aporta_a"], data["naturaleza"]
        )
        return jsonify({
            "status": "success",
            "message": f"Tool '{data['nombre']}' registrado/actualizado."
        }), 200
    except Exception as e:
        logger.exception("Error al registrar tool '%s': %s", data["nombre"], e)
        return jsonify({
            "status": "error",
            "message": "Error interno al registrar el tool."
        }), 500


@app.route("/api/harmony/pid/reset", methods=["POST"])
def reset_pid():
    """Resetea los acumuladores del controlador PID."""
    try:
        if hasattr(controller_state, "pid_controller"):
            controller_state.pid_controller.reset()
            return jsonify({
                "status": "success", "message": "PID reiniciado."
            }), 200
        else:
            return jsonify({
                "status": "error", "message": "Controlador PID no encontrado."
            }), 500
    except Exception as e:
        logger.exception("Error inesperado al reiniciar PID: %s", e)
        return jsonify({
            "status": "error", "message": "Error interno del servidor."
        }), 500


def main():
    """Función principal para iniciar el servicio."""
    control_thread = threading.Thread(
        target=harmony_control_loop, daemon=True, name="HarmonyControlLoop"
    )
    control_thread.start()
    port = int(os.environ.get("HC_PORT", 7000))
    logger.info(
        "Iniciando servidor Flask para Harmony Controller en puerto %d...",
        port
    )
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
