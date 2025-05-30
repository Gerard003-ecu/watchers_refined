#!/usr/bin/env python3
"""
monitor.py

Este script actúa como el "cogniboard", la torre de control del ecosistema watchers.
Monitorea los endpoints de salud de los servicios, calcula una métrica de entropía 
(definida como el número de servicios que no responden), aplica un controlador PID
para generar una señal de control y envía dicha señal a agent_ai para que realice
acciones correctivas.

Además, este script expone un endpoint de salud (/api/health) mediante Flask para que podman
pueda verificar el estado del servicio.

Variables de entorno:
    CONTROL_INTERVAL: Intervalo (en segundos) entre cada ciclo de monitoreo (default: 30)
    PID_KP, PID_KI, PID_KD: Parámetros del controlador PID (default: 1.0, 0.1, 0.05)
    IDEAL_ENTROPY: Valor deseado de entropía (idealmente 0, indicando un sistema sin fallos)
"""

import os
import time
import requests
from simple_pid import PID
from flask import Flask, jsonify

# Configuración de parámetros a partir de variables de entorno
CONTROL_INTERVAL = float(os.getenv("CONTROL_INTERVAL", "30"))
PID_KP = float(os.getenv("PID_KP", "1.0"))
PID_KI = float(os.getenv("PID_KI", "0.1"))
PID_KD = float(os.getenv("PID_KD", "0.05"))
IDEAL_ENTROPY = float(os.getenv("IDEAL_ENTROPY", "0.0"))

# Definición de los endpoints de salud de los servicios a monitorear
SERVICES = {
    "control": "http://control:7000/api/health",
    "ecu": "http://ecu:8000/api/health",
    "watchers_wave": "http://watchers_wave:5000/api/health",
    "watcher_focus": "http://watcher_focus:6000/api/health",
    "optical_controller": "http://optical_controller:8001/api/health",
    "dashboard": "http://dashboard:8050/api/health",
    "agent_ai": "http://agent_ai:9000/api/health"
}

# Endpoint para enviar la señal de control a agent_ai
AGENT_AI_CONTROL_ENDPOINT = "http://agent_ai:9000/api/control"

# Crear una aplicación Flask para exponer el endpoint de salud del cogniboard
app = Flask(__name__)

@app.route("/api/health", methods=["GET"])
def health():
    """Endpoint de salud para el cogniboard."""
    return jsonify({"status": "success", "module": "cogniboard", "mensaje": "Cogniboard operativo"}), 200

def fetch_health(url):
    """Realiza una solicitud GET al endpoint de salud y retorna True si responde correctamente."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return True
    except Exception:
        return False

def calculate_entropy():
    """
    Calcula la entropía del sistema como el número de servicios caídos.
    Una entropía de 0 indica que todos los servicios están operativos.
    """
    failures = 0
    for name, url in SERVICES.items():
        if not fetch_health(url):
            failures += 1
    return float(failures)

def send_control_signal(signal):
    """
    Envía la señal de control a agent_ai a través del endpoint /api/control.
    """
    try:
        payload = {"control_signal": signal}
        response = requests.post(AGENT_AI_CONTROL_ENDPOINT, json=payload, timeout=5)
        response.raise_for_status()
        print(f"[Cogniboard] Señal enviada: {signal}")
    except Exception as e:
        print(f"[Cogniboard] Error al enviar señal: {e}")

def monitor_loop():
    """
    Bucle principal de monitoreo:
      1. Calcula la entropía del sistema (número de servicios caídos).
      2. Usa un controlador PID para calcular la señal de control basándose en la diferencia entre la entropía medida y la deseada (IDEAL_ENTROPY).
      3. Envía la señal de control a agent_ai.
      4. Espera el intervalo definido antes de repetir el ciclo.
    """
    pid = PID(PID_KP, PID_KI, PID_KD, setpoint=IDEAL_ENTROPY)
    pid.output_limits = (-100, 100)  # Limitar la señal de control

    while True:
        entropy = calculate_entropy()
        control_signal = pid(entropy)
        print(f"[Cogniboard] Entropía: {entropy}, Señal PID: {control_signal}")
        send_control_signal(control_signal)
        time.sleep(CONTROL_INTERVAL)

if __name__ == "__main__":
    # Ejecutar el monitor en un hilo y lanzar el servidor Flask para el healthcheck.
    import threading
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    app.run(host="0.0.0.0", port=9100)
