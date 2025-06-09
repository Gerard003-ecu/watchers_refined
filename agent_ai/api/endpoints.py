#!/usr/bin/env python3
# --- START OF FILE agent_ai/api/endpoints.py ---
"""
endpoints.py - API REST para AgentAI (Núcleo Estratégico)

Permite interactuar con la capa estratégica del ecosistema Watchers.
"""

from flask import Flask, request, jsonify
from utils.logger import get_logger
from agent_ai import agent_ai_instance_app

# import logging # No se usa directamente si usamos get_logger
import os

app = Flask(__name__)
logger = get_logger()  # Usar el logger central configurado

logger.info("API Endpoints de AgentAI cargados.")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Retorna la vista completa del estado estratégico de AgentAI."""
    try:
        estado = agent_ai_instance_app.obtener_estado_completo()
        return jsonify({"status": "success", "data": estado}), 200
    except Exception:
        logger.exception("Error al obtener estado completo de AgentAI")
        return (
            jsonify(
                {
                    "status": "error",
                    "mensaje": "Error interno al obtener estado",
                }
            ),
            500,
        )


@app.route("/api/command", methods=["POST"])
def post_command():
    """
    Recibe un COMANDO ESTRATÉGICO para AgentAI.
    Ejemplos: set_strategy, set_target_setpoint_vector, enable_tool, etc.
    JSON esperado: {"comando": "nombre_comando", "valor": valor_comando}
    """
    data = request.get_json() or {}
    comando = data.get("comando")
    valor = data.get("valor")  # Valor puede ser opcional

    if not comando:
        logger.warning("Solicitud a /api/command sin 'comando'")
        return (
            jsonify(
                {"status": "error", "mensaje": "Falta el campo 'comando'"}
            ),
            400,
        )

    try:
        # Delegar el procesamiento al método de comandos estratégicos
        resultado = agent_ai_instance_app.actualizar_comando_estrategico(
            comando, valor
        )
        # Determinar código de estado basado en el resultado
        status_code = 200 if resultado.get("status") == "success" else 400
        return jsonify(resultado), status_code
    except Exception:
        logger.exception(
            "Error al procesar comando estratégico: %s", comando
        )
        return (
            jsonify({
                "status": "error",
                "mensaje": f"Error interno al procesar comando '{comando}'"
            }),
            500,
        )


@app.route("/api/register", methods=["POST"])
def register_module():
    """
    Registra un nuevo watcher_tool.
    JSON esperado: {"nombre": str, "url": str, "tipo": str (opc), ...}
    La URL debe apuntar al endpoint de salud/estado del módulo.
    """
    data = request.get_json() or {}
    try:
        # La validación detallada ocurre dentro de agent_ai_instance_app
        resultado = agent_ai_instance_app.registrar_modulo(data)
        status_code = 200 if resultado.get("status") == "success" else 400
        return jsonify(resultado), status_code
    except Exception:
        logger.exception("Error al registrar módulo")
        return (
            jsonify(
                {
                    "status": "error",
                    "mensaje": "Error interno al registrar módulo",
                }
            ),
            500,
        )


@app.route("/api/health", methods=["GET"])
def health():
    """Retorna el estado de salud básico de AgentAI."""
    try:
        # Verificar si la instancia y el hilo existen y están vivos
        is_loop_alive = (
            hasattr(agent_ai_instance_app, "_strategic_thread")
            and agent_ai_instance_app._strategic_thread is not None
            and agent_ai_instance_app._strategic_thread.is_alive()
        )

        health_status = "success" if is_loop_alive else "error"
        message = (
            "AgentAI operativo."
            if is_loop_alive
            else "Error: Bucle estratégico no está activo."
        )
        status_code = 200 if is_loop_alive else 503  # Service Unavailable

        return (
            jsonify(
                {
                    "status": health_status,
                    "module": "AgentAI",
                    "message": message,
                    "strategic_loop_active": is_loop_alive,
                }
            ),
            status_code,
        )
    except Exception:
        logger.exception("Error al verificar salud de AgentAI")
        return (
            jsonify(
                {
                    "status": "error",
                    "mensaje": "Error interno al verificar salud",
                }
            ),
            500,
        )


@app.route("/api/control", methods=["POST"])
def control_input():
    """
    Recibe la señal de control externa (ej. desde cogniboard).
    JSON esperado: {"control_signal": any}
    """
    data = request.get_json() or {}
    control_signal = data.get("control_signal")

    if control_signal is None:
        logger.warning("Solicitud a /api/control sin 'control_signal'")
        return (
            jsonify({"status": "error", "mensaje": "Falta 'control_signal'"}),
            400,
        )

    try:
        logger.info(f"Recibida señal de control externa: {control_signal}")
        agent_ai_instance_app.recibir_control_cogniboard(
            control_signal
        )  # Pasar la señal
        return (
            jsonify(
                {"status": "success", "mensaje": "Señal de control recibida"}
            ),
            200,
        )
    except Exception:
        logger.exception("Error al procesar señal de control externa")
        return (
            jsonify(
                {
                    "status": "error",
                    "mensaje": "Error interno al procesar señal de control",
                }
            ),
            500,
        )


@app.route("/api/config", methods=["POST"])
def config_input():
    """
    Recibe el estado de configuración externa (ej. desde config_agent).
    JSON esperado: {"config_status": any}
    """
    data = request.get_json() or {}
    config_status = data.get("config_status")

    if config_status is None:
        logger.warning("Solicitud a /api/config sin 'config_status'")
        return (
            jsonify({"status": "error", "mensaje": "Falta 'config_status'"}),
            400,
        )

    try:
        logger.info(f"Recibido estado de configuración: {config_status}")
        agent_ai_instance_app.recibir_config_status(
            config_status
        )  # Pasar el estado
        return (
            jsonify(
                {
                    "status": "success",
                    "mensaje": "Estado de configuración recibido",
                }
            ),
            200,
        )
    except Exception:
        logger.exception(
            "Error al procesar estado de configuración externa"
        )
        return (
            jsonify({
                "status": "error",
                "mensaje": "Error interno al procesar estado de configuración"
            }),
            500,
        )


# --- Punto de Entrada (si se ejecuta como script) ---
if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 9000)
    )  # Usar PORT genérico si AGENT_AI_PORT no está
    logger.info(
        f"Iniciando servidor Flask para AgentAI API en puerto {port}..."
    )

    # Iniciar el bucle estratégico de AgentAI si no se está ejecutando
    # Esto es importante si se ejecuta este script directamente
    if not agent_ai_instance_app._strategic_thread.is_alive():
        logger.info(
            "Iniciando bucle estratégico de AgentAI desde endpoints.py..."
        )
        agent_ai_instance_app.start_loop()

    # Para producción, usa un servidor WSGI como Gunicorn o Waitress
    # Ejemplo con Waitress:
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=port)
    app.run(
        host="0.0.0.0", port=port, debug=False, use_reloader=False
    )  # Mantener para desarrollo

    # Limpieza al salir (si se ejecuta directamente)
    logger.info("Deteniendo AgentAI API...")
    agent_ai_instance_app.shutdown()
    logger.info("AgentAI API finalizado.")


# --- END OF FILE agent_ai/api/endpoints.py ---
