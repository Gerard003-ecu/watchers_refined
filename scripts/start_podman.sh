########## run_podman #########

#!/bin/bash
# Script optimizado para ejecutar el ecosistema watchers con Podman
# Mejoras:
# - Corrección de rutas de build y contexto
# - Verificación de imágenes existentes antes de construir
# - Uso de caché para reducir tiempos de construcción
# - Mejor manejo de errores y logs
# - Añadida espera básica para dependencias (mejorable con healthchecks)
# - Añadidas variables de entorno faltantes
# - Eliminado volumen pip-cache (usar caché de build)

# --- Configuración ---
PROJECT_NAME="mi-proyecto" # Cambiado para coincidir con compose (afecta nombres de imagen/red)
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/podman_run_$(date +%Y%m%d_%H%M%S).log"
NETWORK_NAME="${PROJECT_NAME}_default" # Usar nombre similar a compose

# --- Crear directorio de logs si no existe ---
mkdir -p "$LOG_DIR"

# --- Función para registrar mensajes en el log ---
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# --- Función para verificar y construir una imagen ---
build_if_needed() {
  local service_name=$1
  local dockerfile_path=$2
  local context_path=$3
  # Usar nombre de imagen como localhost/<project>_<service>:latest para consistencia
  local image_name="localhost/${PROJECT_NAME}_${service_name}:latest"
  local short_image_name="${PROJECT_NAME}_${service_name}" # Para logs

  log "Verificando imagen ${short_image_name} (${image_name})..."
  # Comprobar si la imagen existe localmente
  if ! podman image exists "${image_name}"; then
    log "Construyendo imagen ${short_image_name} desde ${context_path}..."
    # Asegurarse que el contexto existe
    if [ ! -d "${context_path}" ]; then
        log "ERROR: El directorio de contexto '${context_path}' no existe."
        return 1
    fi
     # Asegurarse que el Dockerfile existe
    if [ ! -f "${dockerfile_path}" ]; then
        log "ERROR: El Dockerfile '${dockerfile_path}' no existe."
        return 1
    fi

    # Ejecutar build
    podman build --format docker -f "${dockerfile_path}" -t "${image_name}" "${context_path}" || { # <-- CORRECTO
      log "ERROR: Falló la construcción de ${short_image_name}"
      return 1
    }
    log "Imagen ${short_image_name} construida exitosamente"
  else
    log "Imagen ${short_image_name} ya existe, omitiendo construcción"
  fi
  return 0
}

# --- Función para iniciar un contenedor ---
# Uso: start_container <service_name> <port_mapping> [array_de_env_vars...]
start_container() {
  local service_name=$1
  local port_mapping=$2
  shift 2 # Quitar los dos primeros args, el resto son env vars
  local env_vars_array=("$@") # Capturar el resto como array

  local image_name="localhost/${PROJECT_NAME}_${service_name}:latest"
  local container_name="${PROJECT_NAME}_${service_name}_1" # Nombre similar a compose

  # Detener y eliminar el contenedor si ya existe
  if podman container exists "${container_name}"; then
    log "Deteniendo y eliminando contenedor ${container_name} existente..."
    podman stop "${container_name}" >/dev/null 2>&1
    podman rm "${container_name}" >/dev/null 2>&1
  fi

  log "Iniciando contenedor ${service_name} (${container_name})..."

  # Construir comando de inicio usando un array para seguridad
  local podman_cmd=("podman" "run" "-d" "--name" "${container_name}" "--network" "${NETWORK_NAME}")

  # Agregar mapeo de puertos si existe
  if [[ -n "$port_mapping" ]]; then
    podman_cmd+=("-p" "${port_mapping}")
  fi

  # Agregar volumen para logs (montaje de host)
  # Asegurarse que el directorio de logs del host existe
  mkdir -p "$LOG_DIR"
  # Usar ruta absoluta para el montaje del host es más robusto
  local abs_log_dir
  abs_log_dir=$(readlink -f "$LOG_DIR")
  podman_cmd+=("-v" "${abs_log_dir}:/app/logs:Z") # Añadir :Z para SELinux si es necesario

  # Agregar variables de entorno
  for env_var in "${env_vars_array[@]}"; do
    podman_cmd+=("-e" "${env_var}")
  done

  # Agregar imagen
  podman_cmd+=("${image_name}")

  # Ejecutar comando
  log "Ejecutando: ${podman_cmd[*]}" # Muestra el comando que se ejecutará
  "${podman_cmd[@]}" || {
    log "ERROR: Falló el inicio de ${service_name}"
    # Intentar obtener logs del contenedor fallido
    podman logs "${container_name}" >> "$LOG_FILE" 2>&1
    return 1
  }

  log "Contenedor ${service_name} (${container_name}) iniciado exitosamente"
  return 0
}

# --- Función para esperar a que un servicio esté saludable (básico) ---
wait_for_service() {
    local service_to_wait_for=$1
    local container_name="${PROJECT_NAME}_${service_to_wait_for}_1"
    local max_retries=12 # Esperar máx 60 segundos (12 * 5s)
    local count=0
    log "Esperando a que el servicio ${service_to_wait_for} esté listo..."
    while [[ $count -lt $max_retries ]]; do
        # Podman < 4.x puede no tener healthcheck status directo fácil
        # Comprobación básica: ¿está corriendo?
        if podman container inspect --format '{{.State.Running}}' "${container_name}" 2>/dev/null | grep -q "true"; then
             # Intenta comprobar health si está definido (puede fallar en versiones antiguas o si no hay healthcheck)
             local health_status
             health_status=$(podman container inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}unknown{{end}}' "${container_name}" 2>/dev/null)
             log "Estado de ${service_to_wait_for}: Running, Salud: ${health_status:-'no disponible'}"
             if [[ "$health_status" == "healthy" ]] || [[ "$health_status" == "unknown" ]]; then # Considerar unknown como OK si no hay healthcheck
                 log "Servicio ${service_to_wait_for} parece estar listo."
                 return 0
             fi
        else
             log "Servicio ${service_to_wait_for} aún no está corriendo..."
        fi
        count=$((count + 1))
        sleep 5
    done
    log "ERROR: Timeout esperando a que ${service_to_wait_for} esté listo."
    return 1
}


# === Inicio del script principal ===
log "Iniciando despliegue del ecosistema ${PROJECT_NAME}..."

# --- Crear red si no existe ---
if ! podman network exists "${NETWORK_NAME}"; then
  log "Creando red ${NETWORK_NAME}..."
  podman network create "${NETWORK_NAME}" || {
    log "ERROR: No se pudo crear la red ${NETWORK_NAME}"
    exit 1
  }
else
    log "Red ${NETWORK_NAME} ya existe."
fi

# --- Construir imágenes si es necesario (RUTAS CORREGIDAS) ---
build_if_needed "watchers_wave" "./watchers/watchers_tools/watchers_wave/Dockerfile" "./watchers/watchers_tools/watchers_wave" || exit 1
build_if_needed "watcher_focus" "./watchers/watchers_tools/watcher_focus/Dockerfile" "./watchers/watchers_tools/watcher_focus" || exit 1
build_if_needed "malla_watcher" "./watchers/watchers_tools/malla_watcher/Dockerfile" "./watchers/watchers_tools/malla_watcher" || exit 1
build_if_needed "ecu" "./ecu/Dockerfile" "./ecu" || exit 1 # RUTA CORREGIDA
build_if_needed "harmony_controller" "./control/Dockerfile" "./control" || exit 1 # RUTA CORREGIDA
build_if_needed "agent_ai" "./agent_ai/Dockerfile" "./agent_ai" || exit 1 # RUTA CORREGIDA

# --- Iniciar contenedores en orden de dependencia (aproximado) ---

log "--- Iniciando servicios base ---"
start_container "ecu" "8000:8000" \
    "PYTHONPATH=/app" \
    "PORT=8000" || exit 1
wait_for_service "ecu" || exit 1

start_container "watchers_wave" "5000:5000" \
    "PYTHONPATH=/app" \
    "PORT=5000" || exit 1
wait_for_service "watchers_wave" || exit 1 # Esperar aunque no sea dependencia directa estricta

start_container "watcher_focus" "6000:6000" \
    "PYTHONPATH=/app" \
    "PORT=6000" || exit 1
wait_for_service "watcher_focus" || exit 1

start_container "malla_watcher" "5001:5001" \
    "PYTHONPATH=/app" \
    "MATRIZ_ECU_URL=http://ecu:8000" \
    "TORUS_NUM_CAPAS=3" \
    "TORUS_NUM_FILAS=4" \
    "TORUS_NUM_COLUMNAS=5" \
    "MW_BASE_T=0.6" \
    "MW_BASE_E=0.1" \
    "MW_K_GAIN_T=0.1" \
    "MW_K_GAIN_E=0.05" \
    "MW_INFLUENCE_THRESHOLD=5.0" \
    "MW_MAX_AMPLITUDE_NORM=20.0" \
    "MW_REQUESTS_TIMEOUT=2.0" \
    "MW_SIM_INTERVAL=0.5" \
    "PORT=5001" || exit 1
wait_for_service "malla_watcher" || exit 1

log "--- Iniciando servicios de control ---"
# Variable de entorno compleja para harmony_controller (multilínea para claridad)
HC_WATCHERS_CONFIG=$(cat <<-EOF
{
  "watchers_wave": "http://watchers_wave:5000",
  "watcher_focus": "http://watcher_focus:6000",
  "malla_watcher": "http://malla_watcher:5001"
}
EOF
)

start_container "harmony_controller" "7000:7000" \
    "PYTHONPATH=/app" \
    "ECU_API_URL=http://ecu:8000/api/ecu" \
    "WATCHERS_TOOLS_CONFIG=${HC_WATCHERS_CONFIG}" \
    "PORT=7000" || exit 1
wait_for_service "harmony_controller" || exit 1

log "--- Iniciando servicio estratégico ---"
start_container "agent_ai" "9000:9000" \
    "PYTHONPATH=/app" \
    "HARMONY_CONTROLLER_URL=http://harmony_controller:7000" \
    "AA_INTERVAL=5.0" \
    "AA_REQUESTS_TIMEOUT=4.0" \
    "AA_INITIAL_SETPOINT_VECTOR=[1.0, 0.0]" \
    "AA_INITIAL_STRATEGY=default" \
    "AA_GLOBAL_REQ_PATH=/app/requirements.txt" \
    "AA_INITIAL_WAIT=15.0" \
    "PORT=9000" || exit 1
# No necesitamos esperar a agent_ai usualmente

log "--- Despliegue completado ---"
log "Todos los servicios iniciados correctamente."
log "Para verificar el estado de los contenedores, ejecute: podman ps -a --filter network=${NETWORK_NAME}"
log "Para ver los logs de un contenedor, ejecute: podman logs <nombre_contenedor>"
log "Ejemplo: podman logs ${PROJECT_NAME}_ecu_1"

exit 0