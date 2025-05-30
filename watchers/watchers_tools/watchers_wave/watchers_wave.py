# --- START OF FILE watchers_wave.py (AJUSTADO) ---

#!/usr/bin/env python3
"""
watchers_wave: Simulación (oscilador 2D) con retroalimentación activa y reacción a lambda_foton.
Incluye SonicHarmonizer para ondas ultrasónicas.

Endpoints:
- /api/health (GET): Estado de salud.
- /api/state (GET): Estado actual relevante para control (posición, velocidad, params).
- /api/control (POST): Recibir 'control_signal' y ajustar amortiguamiento 'c_current'.
- /api/malla (GET): estado simulado de la malla (obsoleto/ejemplo).
- /api/wave_foton (POST): actualizar lambda_foton.
- /api/acoustic (GET/POST): estado y control de ondas ultrasónicas.
"""

import json
import math
import logging
import time
import threading
import os
from flask import Flask, request, jsonify
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("watchers_wave") # Usar logger específico

# Parámetros base para el oscilador 2D
C_BASE = 0.2 # Amortiguamiento base
K_GAIN_C = 0.05 # Ganancia para ajustar 'c' desde control_signal
OMEGA_BASE = 2.0 # Frecuencia angular base
LAMBDA_DEFAULT = 600.0 # Lambda por defecto
FACTOR_LAMBDA = 0.01 # Influencia de lambda en omega

# Estado del oscilador 2D (protegido por lock)
state = {
    "t": 0.0,
    "x": 1.0,
    "y": 0.0,
    "vx": 0.0,
    "vy": 1.0,
    "c_current": C_BASE, # Amortiguamiento actual
    "lambda_foton_current": LAMBDA_DEFAULT, # Lambda actual
    "omega_local": OMEGA_BASE # Omega actual (calculado)
}
state_lock = threading.Lock()

# Parámetros para SonicHarmonizer (ultrasonido)
FREQ_BASE = 20000.0
AMPLITUD_BASE = 0.5
# Usar lock si se modifica desde múltiples hilos (ej. API y simulación)
acoustic_state = {"freq_current": FREQ_BASE}
acoustic_lock = threading.Lock()


OUTPUT_FILE = "./monitor_test.txt"
# Crear directorio si no existe
log_dir = os.path.dirname(OUTPUT_FILE)
if log_dir and not os.path.exists(log_dir):
     os.makedirs(log_dir, exist_ok=True)
# Limpiar archivo al inicio
with open(OUTPUT_FILE, "w") as f:
    pass

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

app = Flask(__name__)

# --- Endpoint de Salud ---
@app.route('/api/health', methods=['GET'])
def health():
    # Podría verificar si el hilo de simulación está vivo
    sim_alive = any(t.name == "WaveSimLoop" and t.is_alive() for t in threading.enumerate())
    status = "success" if sim_alive else "warning"
    message = "Módulo Watchers_wave operativo." if sim_alive else "Advertencia: Hilo de simulación no activo."
    return jsonify({
        "status": status,
        "module": "Watchers_wave",
        "message": message,
        "simulation_running": sim_alive
    }), 200 if status == "success" else 503

# --- NUEVO: Endpoint de Estado para Harmony Controller ---
@app.route('/api/state', methods=['GET'])
def get_wave_state():
    """Devuelve el estado actual relevante para el control."""
    with state_lock:
        # Devolver una copia de los datos relevantes
        state_data = {
            "position": {"x": state["x"], "y": state["y"]},
            "velocity": {"vx": state["vx"], "vy": state["vy"]},
            "damping_current": state["c_current"],
            "lambda_foton": state["lambda_foton_current"],
            "omega_local": state["omega_local"],
            "amplitude": math.sqrt(state["x"]**2 + state["y"]**2) # Calcular amplitud actual
        }
    logger.debug(f"Devolviendo estado: {state_data}")
    return jsonify({"status": "success", "state": state_data})

# --- NUEVO: Endpoint de Control para Harmony Controller ---
@app.route('/api/control', methods=['POST'])
def set_wave_control():
    """Recibe una señal de control y ajusta el amortiguamiento 'c_current'."""
    data = request.get_json()
    if not data or "control_signal" not in data:
        logger.error("Solicitud a /api/control sin 'control_signal'")
        return jsonify({"status": "error", "message": "Falta 'control_signal'"}), 400

    try:
        # La señal de control ajustará el amortiguamiento
        # Una señal positiva podría indicar "reducir oscilación" -> aumentar amortiguamiento
        # Una señal negativa podría indicar "aumentar oscilación" -> reducir amortiguamiento
        control_signal = float(data["control_signal"])
        # Ajustar la fórmula según la semántica deseada para control_signal
        # Ejemplo: c_new = C_BASE + K_GAIN_C * control_signal
        # Si control > 0, c aumenta. Si control < 0, c disminuye.
        c_new = C_BASE + K_GAIN_C * control_signal
        c_adjusted = max(c_new, 0) # Evitar amortiguamiento negativo

        with state_lock:
            state["c_current"] = c_adjusted

        logger.info(f"[Control] control_signal={control_signal:.3f} -> c_current ajustado a {c_adjusted:.3f}")
        return jsonify({"status": "success", "message": f"Amortiguamiento ajustado a {c_adjusted:.3f}", "c_current": c_adjusted}), 200

    except (ValueError, TypeError) as e:
        logger.error(f"Error al procesar control_signal: {e} - Data: {data}")
        return jsonify({"status": "error", "message": "control_signal debe ser un número"}), 400
    except Exception as e:
        logger.exception("Error al aplicar señal de control a WatchersWave")
        return jsonify({"status": "error", "message": "Error interno al aplicar control"}), 500

# --- Endpoints Existentes (Revisados) ---

@app.route('/api/wave_foton', methods=['POST'])
def wave_foton():
    """Actualiza lambda_foton."""
    data = request.get_json() or {}
    try:
        new_lambda = float(data.get("lambda_foton", LAMBDA_DEFAULT))
        with state_lock:
            state["lambda_foton_current"] = new_lambda
            # Recalcular omega_local inmediatamente si depende de lambda
            state["omega_local"] = get_omega_local(new_lambda)
        logger.info(f"[Foton] lambda_foton_current actualizado a {new_lambda:.2f}, omega_local recalculado a {state['omega_local']:.3f}")
        return jsonify({"status": "success", "lambda_foton_current": new_lambda}), 200
    except (ValueError, TypeError):
         logger.error(f"Valor inválido para lambda_foton: {data.get('lambda_foton')}")
         return jsonify({"status": "error", "message": "lambda_foton debe ser un número"}), 400

@app.route('/api/malla', methods=['GET'])
def get_malla():
    """Endpoint obsoleto/ejemplo, devuelve amplitud actual."""
    with state_lock:
        amplitude = math.sqrt(state["x"]**2 + state["y"]**2)
    response_data = {
        "status": "success",
        "current_amplitude": amplitude,
        "lambda_foton": state["lambda_foton_current"] # Incluir lambda actual
    }
    return jsonify(response_data), 200

@app.route('/api/acoustic', methods=['GET', 'POST'])
def acoustic():
    """Controla el componente de ultrasonido."""
    if request.method == 'POST':
        data = request.get_json() or {}
        try:
            new_freq = float(data.get("frecuencia", FREQ_BASE))
            if not (20000 <= new_freq <= 100000): # Rango ultrasónico típico
                return jsonify({"status": "error", "message": "Frecuencia fuera del rango ultrasónico (20kHz-100kHz)"}), 400
            with acoustic_lock:
                acoustic_state["freq_current"] = new_freq
            logger.info(f"[Acoustic] freq_current ajustada a {new_freq:.2f} Hz")
            return jsonify({"status": "success", "frecuencia": new_freq}), 200
        except (ValueError, TypeError):
             logger.error(f"Valor inválido para frecuencia: {data.get('frecuencia')}")
             return jsonify({"status": "error", "message": "frecuencia debe ser un número"}), 400
    else: # GET
        with acoustic_lock:
            freq = acoustic_state["freq_current"]
        response_data = {
            "status": "success",
            "frecuencia": freq,
            "amplitud": AMPLITUD_BASE # Amplitud base fija por ahora
        }
        return jsonify(response_data), 200

# --- Lógica de Simulación ---

def get_omega_local(lambda_foton):
    """Calcula omega local basado en lambda."""
    # Asegurar que lambda no sea el default exacto para evitar división por cero si FACTOR_LAMBDA es grande
    # O ajustar la fórmula para que sea más estable
    # Ejemplo: usar diferencia relativa o clamping
    lambda_diff = lambda_foton - LAMBDA_DEFAULT
    omega = OMEGA_BASE + FACTOR_LAMBDA * lambda_diff
    return max(0.1, omega) # Evitar omega cero o negativo

def derivatives(t, current_x, current_y, current_vx, current_vy, omega_local, c_local):
    """Calcula las derivadas del oscilador 2D amortiguado."""
    dxdt = current_vx
    dydt = current_vy
    dvxdt = -(omega_local**2) * current_x - c_local * current_vx
    dvydt = -(omega_local**2) * current_y - c_local * current_vy
    return dxdt, dydt, dvxdt, dvydt

def rk4_step(t, current_x, current_y, current_vx, current_vy, dt, omega_local, c_local):
    """Un paso de integración RK4."""
    dx1, dy1, dvx1, dvy1 = derivatives(t, current_x, current_y, current_vx, current_vy, omega_local, c_local)
    x2, y2, vx2, vy2 = current_x + dx1*dt/2, current_y + dy1*dt/2, current_vx + dvx1*dt/2, current_vy + dvy1*dt/2

    dx2, dy2, dvx2, dvy2 = derivatives(t + dt/2, x2, y2, vx2, vy2, omega_local, c_local)
    x3, y3, vx3, vy3 = current_x + dx2*dt/2, current_y + dy2*dt/2, current_vx + dvx2*dt/2, current_vy + dvy2*dt/2

    dx3, dy3, dvx3, dvy3 = derivatives(t + dt/2, x3, y3, vx3, vy3, omega_local, c_local)
    x4, y4, vx4, vy4 = current_x + dx3*dt, current_y + dy3*dt, current_vx + dvx3*dt, current_vy + dvy3*dt

    dx4, dy4, dvx4, dvy4 = derivatives(t + dt, x4, y4, vx4, vy4, omega_local, c_local)

    x_new = current_x + (dt/6)*(dx1 + 2*dx2 + 2*dx3 + dx4)
    y_new = current_y + (dt/6)*(dy1 + 2*dy2 + 2*dy3 + dy4)
    vx_new = current_vx + (dt/6)*(dvx1 + 2*dvx2 + 2*dvx3 + dvx4)
    vy_new = current_vy + (dt/6)*(dvy1 + 2*dvy2 + 2*dvy3 + dvy4)
    return x_new, y_new, vx_new, vy_new

def simulate_wave_infinite():
    """Bucle principal de simulación."""
    logger.info("Iniciando simulación infinita de watchers_wave.")
    dt = 0.05 # Paso de tiempo
    amplitude_threshold = 1.5 # Umbral para evento

    while True:
        # Leer estado actual y parámetros de forma segura
        with state_lock:
            t = state["t"]
            current_x = state["x"]
            current_y = state["y"]
            current_vx = state["vx"]
            current_vy = state["vy"]
            current_c = state["c_current"]
            current_lambda = state["lambda_foton_current"]
            # Recalcular omega aquí o leerlo si se actualiza en wave_foton
            current_omega = get_omega_local(current_lambda)
            state["omega_local"] = current_omega # Actualizar estado

        # Calcular siguiente paso
        x_new, y_new, vx_new, vy_new = rk4_step(t, current_x, current_y, current_vx, current_vy, dt, current_omega, current_c)
        t_new = t + dt

        # Actualizar estado global de forma segura
        with state_lock:
            state["t"] = t_new
            state["x"] = x_new
            state["y"] = y_new
            state["vx"] = vx_new
            state["vy"] = vy_new

        # Lógica de eventos (ejemplo)
        amplitude = math.sqrt(x_new**2 + y_new**2)
        if amplitude >= amplitude_threshold:
            try:
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[WaveEvent] amplitude={amplitude:.3f}, t={t_new:.2f}\n")
                logger.info(f"[WaveEvent] Umbral de amplitud superado: {amplitude:.3f} en t={t_new:.2f}")
                # Podríamos resetear o modificar algo aquí si superamos el umbral
            except Exception as e:
                logger.error(f"Error al escribir evento en archivo: {e}")

        # Esperar para simular tiempo real (ajustar dt si es necesario)
        time.sleep(dt)

# --- Ejecución ---
def run_flask():
    """Inicia el servidor Flask."""
    # --- MOVER OBTENCIÓN DE PUERTO AQUÍ ---
    port = int(os.environ.get("WATCHERS_WAVE_PORT", 5000))
    # --------------------------------------
    # Usar 'use_reloader=False' es importante cuando se usa threading con Flask
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    # --- NUEVO: Registrarse con AgentAI ---
    MODULE_NAME = "watchers_wave"
    SERVICE_PORT = int(os.environ.get("WATCHERS_WAVE_PORT", 5000))
    MODULE_URL = f"http://{MODULE_NAME}:{SERVICE_PORT}"
    HEALTH_URL = f"{MODULE_URL}/api/health"
    # Definir afinidad y naturaleza (¡AJUSTAR SEGÚN TU DISEÑO!)
    APORTA_A = "malla_watcher" # ¿Afecta a la malla? ¿O a ECU?
    NATURALEZA = "potenciador" # ¿Aumenta la oscilación? ¿O la reduce (reductor)? ¿O la modula?
    DESCRIPTION = "Simulador de oscilador 2D amortiguado con ultrasonido."

    registration_successful = register_with_agent_ai(
        MODULE_NAME, MODULE_URL, HEALTH_URL, "auxiliar", APORTA_A, NATURALEZA, DESCRIPTION
    )
    if not registration_successful:
        logger.warning(f"El módulo '{MODULE_NAME}' continuará sin registro exitoso en AgentAI.")
    # ------------------------------------
    # Iniciar simulación en un hilo
    sim_thread = threading.Thread(target=simulate_wave_infinite, daemon=True, name="WaveSimLoop")
    sim_thread.start()

    # Iniciar Flask (bloqueante, por eso la simulación va en hilo)
    run_flask()

# --- END OF FILE watchers_wave.py ---