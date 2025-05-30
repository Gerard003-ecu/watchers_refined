# --- START OF FILE watcher_focus.py (AJUSTADO) ---

#!/usr/bin/env python3
"""
watcher_focus: Simulación Van der Pol con retroalimentación activa.
Endpoints:
- /api/health (GET): Estado de salud.
- /api/state (GET): Estado actual relevante para control (x, y, z, params).
- /api/control (POST): Recibir 'control_signal' y ajustar 'MU0_current'.
- /api/focus (GET): Endpoint original (ahora redundante con /api/state).
"""

import os # Ya esta añadida.
import math
import time
import threading
import logging
import requests
import json
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("watcher_focus") # Usar logger específico

# Parámetros base
MU0_BASE = 2.0 # Parámetro de no linealidad base
K_GAIN_MU = 0.1 # Ganancia para ajustar mu desde control_signal
K = 0.5 # Acoplamiento con z
ALPHA = 1.0 # Tasa de relajación de z
BETA = 0.1 # Influencia de x,y en z
THRESHOLD = 1.0 # Umbral para la dinámica de z
Z_TARGET = 0.0 # Valor objetivo para z

# Estado de la simulación (protegido por lock)
current_state = {
    "t": 0.0,
    "x": 1.0, # Posición inicial x
    "y": 0.0, # Posición inicial y (velocidad dx/dt)
    "z": 0.5, # Variable de control interna z
    "phase": 0.0, # Fase calculada
    "z_error": 0.5, # Error de z respecto a Z_TARGET
    "mu0_current": MU0_BASE # Parámetro mu actual
}
state_lock = threading.Lock()

AGENT_AI_REGISTER_URL = os.environ.get("AGENT_AI_REGISTER_URL", "http://agent_ai:9000/api/register")
MAX_REGISTRATION_RETRIES = 5 # Intentar registrarse varias veces al inicio
RETRY_DELAY = 5 # Segundos entre reintentos

def register_with_agent_ai(module_name: str, module_url: str, health_url: str, module_type: str, aporta_a: str, naturaleza: str, description: str = ""):
    """Intenta registrar este módulo con AgentAI, con reintentos."""
    payload = {
        "nombre": module_name,
        "url": module_url, # URL base para control/estado
        "url_salud": health_url, # URL específica de salud
        "tipo": module_type,
        "aporta_a": aporta_a,
        "naturaleza_auxiliar": naturaleza,
        "descripcion": description
        # Podrías añadir más metadata si AgentAI la usa
    }
    logger.info(f"Intentando registrar '{module_name}' en AgentAI ({AGENT_AI_REGISTER_URL})...")
    for attempt in range(MAX_REGISTRATION_RETRIES):
        try:
            response = requests.post(AGENT_AI_REGISTER_URL, json=payload, timeout=4.0)
            response.raise_for_status() # Lanza excepción para errores 4xx/5xx
            if response.status_code == 200:
                 logger.info(f"Registro de '{module_name}' exitoso en AgentAI.")
                 return True # Salir si el registro es exitoso
            else:
                 # Esto no debería ocurrir si raise_for_status funciona, pero por si acaso
                 logger.warning(f"Registro de '{module_name}' recibido con status {response.status_code}. Respuesta: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión al intentar registrar '{module_name}' (intento {attempt + 1}/{MAX_REGISTRATION_RETRIES}): {e}")
        except Exception as e:
             logger.error(f"Error inesperado durante el registro de '{module_name}' (intento {attempt + 1}/{MAX_REGISTRATION_RETRIES}): {e}")

        if attempt < MAX_REGISTRATION_RETRIES - 1:
            logger.info(f"Reintentando registro en {RETRY_DELAY} segundos...")
            time.sleep(RETRY_DELAY)
        else:
            logger.error(f"No se pudo registrar '{module_name}' en AgentAI después de {MAX_REGISTRATION_RETRIES} intentos.")
            return False # Falló después de todos los reintentos
    return False # En caso de que el bucle termine inesperadamente

app_focus = Flask(__name__)

# --- Endpoint de Salud ---
@app_focus.route('/api/health', methods=['GET'])
def health():
    sim_alive = any(t.name == "FocusSimLoop" and t.is_alive() for t in threading.enumerate())
    status = "success" if sim_alive else "warning"
    message = "Módulo Watcher_focus operativo." if sim_alive else "Advertencia: Hilo de simulación no activo."
    return jsonify({
        "status": status,
        "module": "Watcher_focus",
        "message": message,
        "simulation_running": sim_alive
    }), 200 if status == "success" else 503

# --- NUEVO: Endpoint de Estado para Harmony Controller ---
@app_focus.route('/api/state', methods=['GET'])
def get_focus_state():
    """Devuelve el estado actual relevante para el control."""
    with state_lock:
        # Devolver una copia de los datos relevantes
        state_data = {
            "position": {"x": current_state["x"], "y": current_state["y"]},
            "internal_control": {"z": current_state["z"]},
            "phase": current_state["phase"],
            "z_error": current_state["z_error"],
            "mu0_current": current_state["mu0_current"]
        }
    logger.debug(f"Devolviendo estado: {state_data}")
    return jsonify({"status": "success", "state": state_data})

# --- NUEVO: Endpoint de Control para Harmony Controller ---
@app_focus.route('/api/control', methods=['POST'])
def set_focus_control():
    """Recibe una señal de control y ajusta el parámetro 'MU0_current'."""
    data = request.get_json()
    if not data or "control_signal" not in data:
        logger.error("Solicitud a /api/control sin 'control_signal'")
        return jsonify({"status": "error", "message": "Falta 'control_signal'"}), 400

    try:
        # La señal de control ajustará el parámetro mu0
        # Ejemplo: señal positiva -> aumentar mu0 (más oscilación/inestabilidad VdP)
        #          señal negativa -> disminuir mu0 (más amortiguamiento VdP)
        control_signal = float(data["control_signal"])
        # Ajustar fórmula según semántica deseada
        mu0_new = MU0_BASE + K_GAIN_MU * control_signal
        mu0_adjusted = max(mu0_new, 0.0) # Evitar mu negativo

        with state_lock:
            current_state["mu0_current"] = mu0_adjusted

        logger.info(f"[Control] control_signal={control_signal:.3f} -> MU0_current ajustado a {mu0_adjusted:.3f}")
        return jsonify({"status": "success", "message": f"Parámetro MU0 ajustado a {mu0_adjusted:.3f}", "MU0_current": mu0_adjusted}), 200

    except (ValueError, TypeError) as e:
        logger.error(f"Error al procesar control_signal: {e} - Data: {data}")
        return jsonify({"status": "error", "message": "control_signal debe ser un número"}), 400
    except Exception as e:
        logger.exception("Error al aplicar señal de control a WatcherFocus")
        return jsonify({"status": "error", "message": "Error interno al aplicar control"}), 500

# --- Endpoint Original (puede ser redundante ahora) ---
@app_focus.route('/api/focus', methods=['GET'])
def get_focus():
    """Devuelve el estado completo (similar a /api/state pero formato original)."""
    with state_lock:
        state_copy = current_state.copy()
    # Podrías mantenerlo por compatibilidad o redirigir/eliminar
    logger.warning("Llamada a endpoint obsoleto /api/focus, usar /api/state.")
    return jsonify({"status": "success", "focus_state": state_copy}), 200

# --- Lógica de Simulación (Van der Pol con control z) ---

def derivatives(t, x, y, z, current_mu0):
    """Calcula las derivadas del sistema."""
    # El parámetro mu ahora depende de z y del mu0 actual ajustado por el control
    mu = current_mu0 + K * z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    # La dinámica de z intenta llevar x,y al círculo unitario (THRESHOLD=1.0)
    # y z hacia Z_TARGET (0.0)
    dzdt = -ALPHA * (z - Z_TARGET) + BETA * (math.sqrt(x**2 + y**2) - THRESHOLD) # Usar norma en lugar de abs(x)+abs(y)
    return dxdt, dydt, dzdt

def rk4_step(t, x, y, z, dt, current_mu0):
    """Un paso de integración RK4."""
    dx1, dy1, dz1 = derivatives(t, x, y, z, current_mu0)
    x2, y2, z2 = x + dx1*dt/2, y + dy1*dt/2, z + dz1*dt/2

    dx2, dy2, dz2 = derivatives(t + dt/2, x2, y2, z2, current_mu0)
    x3, y3, z3 = x + dx2*dt/2, y + dy2*dt/2, z + dz2*dt/2

    dx3, dy3, dz3 = derivatives(t + dt/2, x3, y3, z3, current_mu0)
    x4, y4, z4 = x + dx3*dt, y + dy3*dt, z + dz3*dt

    dx4, dy4, dz4 = derivatives(t + dt, x4, y4, z4, current_mu0)

    x_new = x + (dt/6)*(dx1 + 2*dx2 + 2*dx3 + dx4)
    y_new = y + (dt/6)*(dy1 + 2*dy2 + 2*dy3 + dy4)
    z_new = z + (dt/6)*(dz1 + 2*dz2 + 2*dz3 + dz4)
    return x_new, y_new, z_new

def update_indicators(t, x, y, z):
    """Calcula indicadores derivados del estado."""
    phase = math.atan2(y, x)
    z_error = abs(z - Z_TARGET)
    return {"t": t, "x": x, "y": y, "z": z, "phase": phase, "z_error": z_error}

def simulate_watcher_focus_infinite(dt=0.01):
    """Bucle principal de simulación."""
    logger.info("Iniciando simulación infinita de watcher_focus (Van der Pol).")
    log_interval_steps = 10 # Loguear cada 10 pasos

    # Leer estado inicial de forma segura
    with state_lock:
        t = current_state["t"]
        x = current_state["x"]
        y = current_state["y"]
        z = current_state["z"]
        mu0 = current_state["mu0_current"]

    step_count = 0
    while True:
        # Leer mu0 actual (puede cambiar por la API)
        with state_lock:
            mu0 = current_state["mu0_current"]

        # Calcular siguiente paso
        x_new, y_new, z_new = rk4_step(t, x, y, z, dt, mu0)
        t_new = t + dt

        # Actualizar estado para el siguiente paso
        x, y, z, t = x_new, y_new, z_new, t_new

        # Actualizar estado global e indicadores periódicamente
        step_count += 1
        if step_count % log_interval_steps == 0:
            indicators = update_indicators(t, x, y, z)
            with state_lock:
                current_state.update(indicators)
                # Asegurarse de que mu0_current también esté actualizado en el estado
                current_state["mu0_current"] = mu0
            logger.info(f"[FocusOsc] t={t:.2f} x={x:.3f} y={y:.3f} z={z:.3f} mu0={mu0:.3f}")

        time.sleep(dt)

# --- Ejecución ---
def run_focus_api():
    """Inicia el servidor Flask."""
    # --- MOVER OBTENCIÓN DE PUERTO AQUÍ ---
    port = int(os.environ.get("WATCHER_FOCUS_PORT", 6000))
    # --------------------------------------
    app_focus.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    # --- NUEVO: Registrarse con AgentAI ---
    MODULE_NAME = "watcher_focus"
    SERVICE_PORT = int(os.environ.get("WATCHER_FOCUS_PORT", 6000))
    MODULE_URL = f"http://{MODULE_NAME}:{SERVICE_PORT}"
    HEALTH_URL = f"{MODULE_URL}/api/health"
    # Definir afinidad y naturaleza (¡AJUSTAR SEGÚN TU DISEÑO!)
    APORTA_A = "malla_watcher" # ¿Afecta a la malla? ¿O a ECU?
    NATURALEZA = "reductor" # ¿Reduce la oscilación? ¿O la aumenta (potenciador)? ¿O la modula?
    DESCRIPTION = "Simulador de oscilador Van der Pol con control interno."

    registration_successful = register_with_agent_ai(
        MODULE_NAME, MODULE_URL, HEALTH_URL, "auxiliar", APORTA_A, NATURALEZA, DESCRIPTION
    )
    if not registration_successful:
        logger.warning(f"El módulo '{MODULE_NAME}' continuará sin registro exitoso en AgentAI.")
    # ------------------------------------
    # Iniciar simulación en hilo
    sim_thread = threading.Thread(target=simulate_watcher_focus_infinite, daemon=True, name="FocusSimLoop")
    sim_thread.start()

    # Iniciar API Flask (bloqueante)
    run_focus_api()

# --- END OF FILE watcher_focus.py ---