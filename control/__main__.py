import logging
import os
import threading

# Importar la app y el bucle de control desde el módulo principal
from .harmony_controller import app, harmony_control_loop

# Configurar un logger básico para este punto de entrada
logger = logging.getLogger(__name__)


def main():
    """
    Punto de entrada principal para ejecutar el servicio Harmony Controller.

    Esta función realiza dos acciones clave:
    1. Inicia el bucle de control principal (`harmony_control_loop`) en un
       hilo de fondo. Este hilo se marca como 'daemon' para que no bloquee
       la finalización del programa.
    2. Inicia el servidor web Flask (`app.run`), que escucha las peticiones
       HTTP en la dirección y puerto especificados.

    La configuración del puerto se obtiene de la variable de entorno HC_PORT,
    con un valor por defecto de 7000.
    """
    # Es buena práctica asegurarse de que el logging esté configurado.
    # harmony_controller.py ya lo configura, pero si este script se ejecutara
    # de alguna manera extraña, esto proporciona un fallback.
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)

    # 1. Iniciar el bucle de control en un hilo separado
    control_thread = threading.Thread(
        target=harmony_control_loop, daemon=True, name="HarmonyControlLoop"
    )
    control_thread.start()

    # 2. Iniciar el servidor Flask
    # El puerto se puede configurar a través de variables de entorno
    port = int(os.environ.get("HC_PORT", 7000))
    logger.info(
        "Iniciando servidor Flask para Harmony Controller en el puerto %d...", port
    )
    # use_reloader=False es importante para producción y para evitar
    # que el código se ejecute dos veces.
    # debug=False es igualmente crucial por seguridad y rendimiento.
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
