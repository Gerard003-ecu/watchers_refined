#!/usr/bin/env python3
# --- START OF FILE agent_ai/api/endpoints.py ---
"""API REST para el Núcleo Estratégico AgentAI.

Este módulo define los endpoints de la API Flask para interactuar con AgentAI,
el componente central de la capa estratégica del ecosistema Watchers.
Proporciona interfaces para obtener el estado, enviar comandos, registrar
módulos y recibir señales de control y configuración.
"""

from flask import Flask, request, jsonify
from ..utils.logger import get_logger
from .. import agent_ai as agent_ai_core

# import logging # No se usa directamente si usamos get_logger
import os

app = Flask(__name__)
logger = get_logger()  # Usar el logger central configurado

logger.info("API Endpoints de AgentAI cargados.")


@app.route("/api/status", methods=["GET"])
def get_status():
    """Obtiene el estado estratégico completo de AgentAI.

    Este endpoint permite a los clientes consultar una vista detallada del
    estado actual de AgentAI, incluyendo información de módulos registrados,
    estrategias activas y otros datos relevantes para la operación.

    Returns:
        flask.Response: Un objeto JSON que contiene:
            - status (str): "success" si la operación fue exitosa, "error" en
              caso contrario.
            - data (dict): Un diccionario con el estado completo de AgentAI si
              la operación fue exitosa.
            - mensaje (str): Un mensaje descriptivo en caso de error.
    """
    try:
        estado = agent_ai_core.agent_ai_instance.obtener_estado_completo()
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


@app.route("/api/metrics", methods=["POST"])
def receive_metrics():
    """
    Recibe métricas de rendimiento de otros servicios.
    """
    data = request.get_json()
    if not data or not all(k in data for k in ["source_service", "function_name", "execution_time", "call_count"]):
        logger.warning("Solicitud a /api/metrics con payload inválido: %s", data)
        return jsonify({"status": "error", "mensaje": "Payload JSON inválido o faltan claves requeridas."}), 400

    try:
        # Delegar el almacenamiento a la instancia de AgentAI
        agent_ai_core.agent_ai_instance.store_metric(data)
        return jsonify({"status": "success", "mensaje": "Métrica recibida."}), 200
    except Exception as e:
        logger.exception("Error al procesar métrica: %s", data)
        return jsonify({"status": "error", "mensaje": f"Error interno al procesar la métrica: {e}"}), 500


@app.route("/api/command", methods=["POST"])
def post_command():
    """Procesa un comando estratégico enviado a AgentAI.

    Este endpoint recibe comandos para influir en el comportamiento de
    AgentAI. Los comandos pueden variar desde establecer una nueva
    estrategia hasta modificar parámetros operativos específicos.

    El cuerpo de la solicitud debe ser un JSON con la siguiente
    estructura:
    {
        "comando": "nombre_del_comando",
        "valor": "valor_asociado_al_comando"
    }

    Args:
        None explícito en la firma, pero espera un JSON
        en el cuerpo de la solicitud.

    Returns:
        flask.Response: Un objeto JSON que contiene:
            - status (str): "success" si el comando fue procesado
              exitosamente, "error" en caso contrario.
            - mensaje (str): Un mensaje descriptivo sobre el resultado de la
              operación.
            - (otros campos): Puede incluir campos adicionales dependiendo del
              resultado del comando específico.
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
        resultado = agent_ai_core.agent_ai_instance.actualizar_comando_estrategico(
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


@app.route("/commands/synchronize_region", methods=["POST"])
def synchronize_region_command():
    """
    Recibe un comando para iniciar una maniobra de sincronización de fase.
    """
    data = request.get_json()
    if not data or "region" not in data or "target_phase" not in data:
        logger.warning(
            "Payload inválido para /commands/synchronize_region: %s", data
        )
        return jsonify({"status": "error", "message": "Payload inválido, se requiere 'region' y 'target_phase'"}), 400

    try:
        region = data["region"]
        target_phase = float(data["target_phase"])

        # Delegar la tarea a la instancia de AgentAI
        agent_ai_core.agent_ai_instance._delegate_phase_synchronization_task(
            region, target_phase
        )

        return (
            jsonify(
                {
                    "status": "command_accepted",
                    "message": f"Sincronización de fase iniciada para la región '{region}'.",
                }
            ),
            202,
        )

    except ValueError:
        logger.warning(
            "Error de valor en /commands/synchronize_region: 'target_phase' no es un flotante válido. Payload: %s",
            data,
        )
        return jsonify({"status": "error", "message": "El campo 'target_phase' debe ser un número."}), 400
    except Exception as e:
        logger.exception("Error al procesar el comando synchronize_region")
        return (
            jsonify(
                {"status": "error", "message": "Error interno al procesar el comando."}
            ),
            500,
        )


@app.route("/api/register", methods=["POST"])
def register_module():
    """
    Registra un nuevo módulo watcher_tool en AgentAI.

    Este endpoint permite a los módulos `watcher_tool` darse de alta en
    AgentAI, proporcionando la información necesaria para que AgentAI pueda
    interactuar con ellos. La URL proporcionada debe apuntar al endpoint de
    salud/estado del módulo que se registra.

    El cuerpo de la solicitud debe ser un JSON con
    la siguiente estructura (ejemplo):
    {
        "nombre": "NombreDelWatcher",
        "url": "http://host.docker.internal:puerto/api/health",
        "tipo": "TipoDeWatcher", // Opcional
        // ... otros campos relevantes para el módulo
    }

    Args:
        None explícito en la firma, pero espera un JSON
        en el cuerpo de la solicitud.

    Returns:
        flask.Response: Un objeto JSON que contiene:
            - status (str): "success" si el módulo fue registrado
              exitosamente, "error" en caso contrario.
            - mensaje (str): Un mensaje descriptivo sobre el resultado del
              registro.
            - module_id (str): El ID asignado al módulo si el registro fue
              exitoso.
    """
    data = request.get_json() or {}
    try:
        # La validación detallada ocurre dentro de agent_ai_instance_app
        resultado = agent_ai_core.agent_ai_instance.registrar_modulo(data)
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
    """Verifica el estado de salud básico de AgentAI.

    Este endpoint proporciona una comprobación rápida para
    determinar si AgentAI está operativo. Verifica principalmente
    si el bucle estratégico principal de AgentAI se está ejecutando.

    Returns:
        flask.Response: Un objeto JSON que contiene:
        - status (str): "success" si AgentAI está saludable, "error" en
          caso contrario.
        - module (str): Siempre "AgentAI".
        - message (str): Un mensaje descriptivo del estado de salud.
        - strategic_loop_active (bool): True si el bucle estratégico está
          activo, False en caso contrario.
        - mensaje (str): (En caso de error interno) Un mensaje descriptivo
          del error.
    """
    try:
        # Verificar si la instancia y el hilo existen y están vivos
        is_loop_alive = (
            hasattr(agent_ai_core, "_strategic_thread")
            and agent_ai_core._strategic_thread is not None
            and agent_ai_core._strategic_thread.is_alive()
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
    """Recibe y procesa una señal de control externa para AgentAI.

    Este endpoint está diseñado para que sistemas externos, como Cogniboard,
    puedan enviar señales de control que influyan en las decisiones o
    el comportamiento de AgentAI.

    El cuerpo de la solicitud debe ser un JSON con la siguiente estructura:
    {
        "control_signal": <valor_de_la_senal_de_control>
    }
    El tipo de `<valor_de_la_senal_de_control>` puede variar según la señal.

    Args:
        None explícito en la firma, pero espera un JSON
        en el cuerpo de la solicitud.

    Returns:
        flask.Response: Un objeto JSON que contiene:
            - status (str): "success" si la señal fue recibida y procesada
              exitosamente, "error" en caso contrario.
            - mensaje (str): Un mensaje descriptivo sobre el resultado de la
              operación.
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
        agent_ai_core.agent_ai_instance.recibir_control_cogniboard(
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


@app.route("/api/config_report", methods=["POST"])
def config_report_input():
    """
    Recibe el informe de configuración completo de config_agent.

    Este endpoint procesa el informe JSON, valida su estructura básica y lo
    pasa a la instancia de AgentAI para su procesamiento y almacenamiento.
    """
    report = request.get_json()
    if not report or not all(k in report for k in ["global_status", "services", "mic_validation"]):
        logger.warning("Solicitud a /api/config_report con payload inválido: %s", report)
        return jsonify({"status": "error", "message": "Payload JSON inválido o faltan claves requeridas."}), 400

    try:
        # Delegar el procesamiento a la instancia de AgentAI
        agent_ai_core.agent_ai_instance.update_system_architecture(report)
        logger.info("Informe de configuración recibido y procesado.")
        return jsonify({"status": "report_received"}), 200
    except Exception as e:
        logger.exception("Error al procesar el informe de configuración: %s", e)
        return jsonify({"status": "error", "message": f"Error interno al procesar el informe: {e}"}), 500


# --- Punto de Entrada (si se ejecuta como script) ---
if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 9000)
    )  # Usar PORT genérico si AGENT_AI_PORT no está
    logger.info(
        f"Iniciando servidor Flask para AgentAI API en puerto {port}..."
    )

    # Iniciar el bucle estratégico de AgentAI.
    logger.info("Iniciando bucle estratégico de AgentAI desde endpoints.py...")
    agent_ai_core.start_loop()

    try:
        # Para producción, se recomienda un servidor WSGI como Gunicorn o Waitress
        # Ejemplo con Waitress:
        # from waitress import serve
        # serve(app, host="0.0.0.0", port=port)
        app.run(
            host="0.0.0.0", port=port, debug=False, use_reloader=False
        )
    finally:
        # Limpieza al salir
        logger.info("Deteniendo AgentAI API...")
        agent_ai_core.shutdown()
        logger.info("AgentAI API finalizado.")


# --- END OF FILE agent_ai/api/endpoints.py ---
