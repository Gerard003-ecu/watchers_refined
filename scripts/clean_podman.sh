############ clean_podman ##############

#!/bin/bash
# Script para limpiar contenedores e imágenes del ecosistema watchers en Podman

# --- Configuración (DEBE COINCIDIR CON run_podman.sh) ---
PROJECT_NAME="mi-proyecto"
NETWORK_NAME="${PROJECT_NAME}_default"
# Lista de nombres de servicios tal como se usan en los nombres de imagen/contenedor
SERVICES=("ecu" "watchers_wave" "watcher_focus" "malla_watcher" "harmony_controller" "agent_ai")

# --- Función para registrar mensajes ---
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# === Inicio del script de limpieza ===
log "Iniciando limpieza del ecosistema ${PROJECT_NAME}..."

# --- 1. Detener Contenedores ---
log "--- Deteniendo contenedores del proyecto ${PROJECT_NAME} ---"
# Obtener IDs de contenedores (corriendo o detenidos) en la red del proyecto
CONTAINER_IDS=$(podman ps -a --filter network=${NETWORK_NAME} -q)

if [[ -z "$CONTAINER_IDS" ]]; then
  log "No se encontraron contenedores activos o detenidos para el proyecto ${PROJECT_NAME} en la red ${NETWORK_NAME}."
else
  log "Contenedores encontrados: $CONTAINER_IDS"
  log "Intentando detener contenedores..."
  # Usar xargs para manejar múltiples IDs de forma segura
  # El comando stop puede fallar si ya están detenidos, por eso no salimos en error aquí
  echo "$CONTAINER_IDS" | xargs --no-run-if-empty podman stop || log "Advertencia: Algunos contenedores no pudieron ser detenidos (quizás ya estaban detenidos)."
  log "Contenedores detenidos (o ya estaban detenidos)."
fi

# --- 2. Eliminar Contenedores ---
log "--- Eliminando contenedores del proyecto ${PROJECT_NAME} ---"
# Volver a obtener IDs por si acaso (aunque los anteriores deberían ser los mismos si stop funcionó)
CONTAINER_IDS=$(podman ps -a --filter network=${NETWORK_NAME} -q)

if [[ -z "$CONTAINER_IDS" ]]; then
  log "No se encontraron contenedores para eliminar."
else
  log "Intentando eliminar contenedores: $CONTAINER_IDS"
  echo "$CONTAINER_IDS" | xargs --no-run-if-empty podman rm || {
    log "ERROR: No se pudieron eliminar todos los contenedores. Verifique manualmente con 'podman ps -a --filter network=${NETWORK_NAME}'"
    exit 1 # Salir si no se pueden eliminar contenedores, ya que bloqueará la eliminación de imágenes
  }
  log "Contenedores eliminados exitosamente."
fi

# --- 3. Eliminar Imágenes ---
log "--- Eliminando imágenes del proyecto ${PROJECT_NAME} ---"
IMAGES_REMOVED=0
IMAGES_FAILED=0

for service in "${SERVICES[@]}"; do
  image_name="localhost/${PROJECT_NAME}_${service}:latest"
  short_image_name="${PROJECT_NAME}_${service}" # Para logs

  log "Verificando imagen ${short_image_name} (${image_name})..."
  # Comprobar si la imagen existe antes de intentar borrarla
  if podman image exists "${image_name}"; then
    log "Intentando eliminar imagen ${image_name}..."
    podman rmi "${image_name}" || {
      log "ERROR: Falló la eliminación de la imagen ${image_name}. Puede estar en uso por otro contenedor o tener etiquetas dependientes."
      IMAGES_FAILED=$((IMAGES_FAILED + 1))
    } && { # Solo contar como eliminada si el comando rmi tuvo éxito
      log "Imagen ${image_name} eliminada exitosamente."
      IMAGES_REMOVED=$((IMAGES_REMOVED + 1))
    }
  else
    log "Imagen ${image_name} no encontrada, omitiendo."
  fi
done

log "--- Resumen de limpieza de imágenes ---"
log "Imágenes eliminadas: ${IMAGES_REMOVED}"
log "Imágenes con fallo al eliminar: ${IMAGES_FAILED}"

if [[ $IMAGES_FAILED -gt 0 ]]; then
    log "ADVERTENCIA: No se pudieron eliminar todas las imágenes. Ejecute 'podman images | grep ${PROJECT_NAME}' para verificar."
    # No salimos con error aquí, la limpieza principal (contenedores) puede haber sido exitosa
fi

log "--- Limpieza completada ---"
exit 0