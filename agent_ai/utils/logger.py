# --- START OF FILE agent_ai/utils/logger.py (AJUSTADO) ---
import logging
import os
import sys

LOG_DIR_ENV = os.environ.get("WATCHERS_LOG_DIR", "logs")
base_dir = os.getcwd()
LOG_DIR = os.path.join(base_dir, LOG_DIR_ENV)

# ... (Definición de LOG_DIR, LOG_FILE) ...
try:
    os.makedirs(LOG_DIR, exist_ok=True) # Ahora LOG_DIR está definido
except OSError as e:
    print(f"ADVERTENCIA: No se pudo crear el directorio de logs '{LOG_DIR}': {e}", file=sys.stderr)

LOG_FILE = os.path.join(LOG_DIR, "agent_ai.log")

logger = logging.getLogger("agent_ai")
# Establecer nivel aquí, pero los handlers controlarán la salida final
logger.setLevel(logging.DEBUG)
# Evitar propagación al logger raíz para control total
logger.propagate = False

# --- Lógica de Configuración Idempotente Refinada ---
def setup_logger_handlers():
    """Configura los handlers si no existen o si son diferentes."""
    global logger # Asegurar que usamos la instancia global
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Verificar si ya existe un FileHandler configurado correctamente
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == LOG_FILE
        for h in logger.handlers
    )

    if not has_file_handler:
        # Remover FileHandlers antiguos si existen (por si acaso)
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
        # Añadir el nuevo
        try:
            file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            # logger.debug("Nuevo File handler añadido.")
        except Exception as e:
             print(f"ADVERTENCIA: No se pudo configurar el file handler para logger en '{LOG_FILE}': {e}", file=sys.stderr)

    # Verificar si ya existe un StreamHandler configurado correctamente
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )

    if not has_console_handler:
        # Remover StreamHandlers antiguos (que no sean FileHandlers)
        for handler in logger.handlers[:]:
             if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                 logger.removeHandler(handler)
        # Añadir el nuevo
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        # logger.debug("Nuevo Console handler añadido.")

# Llamar a la configuración al importar el módulo
setup_logger_handlers()

def get_logger():
    """Retorna el logger configurado."""
    # Opcional: Llamar a setup_logger_handlers() aquí también por si acaso,
    # aunque debería haberse llamado al importar.
    # setup_logger_handlers()
    return logger

# --- Ejemplo de Uso (si se ejecuta directamente) ---
if __name__ == "__main__":
    # Asegurarse de que la configuración se aplique si se ejecuta como script
    # (Aunque normalmente se importa como módulo)
    log = get_logger()
    log.debug("Mensaje DEBUG: Inicio de la aplicación (ejecución directa).")
    log.info("Mensaje INFO: Aplicación en ejecución (ejecución directa).")
    log.warning("Mensaje WARNING: Advertencia de ejemplo (ejecución directa).")
    log.error("Mensaje ERROR: Se ha producido un error (ejecución directa).")
    log.critical("Mensaje CRITICAL: Error crítico (ejecución directa).")