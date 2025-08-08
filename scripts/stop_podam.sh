########## stop_podman ################

#!/bin/bash
# Script para DETENER todos los contenedores del ecosistema watchers en Podman

# --- Configuración (DEBE COINCIDIR CON run_podman.sh) ---
PROJECT_NAME="mi-proyecto"
NETWORK_NAME="${PROJECT_NAME}_default"

# --- Función para registrar mensajes ---
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# === Inicio del script de detención ===
log "Iniciando detención de contenedores del ecosistema ${PROJECT_NAME}..."

# --- Detener Contenedores ---
log "--- Buscando contenedores del proyecto ${PROJECT_NAME} en la red ${NETWORK_NAME} ---"
# Obtener IDs de contenedores (SOLO los que están corriendo)
CONTAINER_IDS=$(podman ps --filter network=${NETWORK_NAME} -q)

if [[ -z "$CONTAINER_IDS" ]]; then
  log "No se encontraron contenedores en ejecución para el proyecto ${PROJECT_NAME}."
  exit 0 # Salir limpiamente si no hay nada que detener
else
  log "Contenedores en ejecución encontrados: $CONTAINER_IDS"
  log "Intentando detener contenedores..."
  # Usar xargs para manejar múltiples IDs de forma segura
  echo "$CONTAINER_IDS" | xargs --no-run-if-empty podman stop || {
      log "Advertencia: No se pudieron detener todos los contenedores limpiamente (puede que alguno ya estuviera parando)."
      # Podríamos intentar forzar aquí si fuera necesario, pero stop suele ser suficiente
  }
  log "Comando de detención enviado a los contenedores."
fi

log "--- Verificación ---"
# Esperar un poco y verificar
sleep 2
REMAINING_CONTAINERS=$(podman ps --filter network=${NETWORK_NAME} -q)
if [[ -z "$REMAINING_CONTAINERS" ]]; then
    log "Todos los contenedores del proyecto han sido detenidos exitosamente."
    exit 0
else
    log "ERROR: Algunos contenedores aún podrían estar corriendo: $REMAINING_CONTAINERS"
    log "Verifique manualmente con 'podman ps --filter network=${NETWORK_NAME}'"
    exit 1
fi