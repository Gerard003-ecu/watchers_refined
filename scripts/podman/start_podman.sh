#!/bin/bash
# ==============================================================================
# Script to Start the Watchers Ecosystem with Podman
#
# Features:
# - Robustness: Uses 'set -euo pipefail'.
# - Idempotent: Cleans up old containers before starting new ones.
# - Healthchecks: Waits for services to be 'healthy' before proceeding.
# - Clear Logging: Color-coded output for better readability.
# - Centralized Config: Key variables are defined at the top.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -euo pipefail

# --- Configuration ---
PROJECT_NAME="watchers"
NETWORK_NAME="${PROJECT_NAME}_default"
LOG_DIR="./logs"
LOG_FILE="${LOG_DIR}/podman_start_$(date +%Y%m%d_%H%M%S).log"

# --- Colors for Logging ---
COLOR_RESET='\033[0m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'

# --- Logging Functions ---
# Usage: log_info "message"
log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1" | tee -a "$LOG_FILE"
}

# Usage: log_success "message"
log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $1" | tee -a "$LOG_FILE"
}

# Usage: log_warn "message"
log_warn() {
    echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $1" | tee -a "$LOG_FILE"
}

# Usage: log_error "message"
log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $1" | tee -a "$LOG_FILE"
}

# --- Main Functions ---

# Creates the log directory if it doesn't exist.
setup_logging() {
    mkdir -p "$LOG_DIR"
    # Create a new log file, or clear an existing one for this run
    > "$LOG_FILE"
    log_info "Logging to ${LOG_FILE}"
}

# Builds a container image if it doesn't already exist.
# Usage: build_if_needed <service_name> <dockerfile_path> <context_path>
build_if_needed() {
    local service_name=$1
    local dockerfile_path=$2
    local context_path=$3
    local image_name="localhost/${PROJECT_NAME}/${service_name}:latest"

    log_info "Checking image for service: ${service_name}..."
    if podman image exists "${image_name}"; then
        log_success "Image '${image_name}' already exists. Skipping build."
    else
        log_info "Building image '${image_name}' from context '${context_path}'..."
        if [ ! -f "${dockerfile_path}" ]; then
            log_error "Dockerfile not found at '${dockerfile_path}'"
            exit 1
        fi
        if [ ! -d "${context_path}" ]; then
            log_error "Build context path not found at '${context_path}'"
            exit 1
        fi

        podman build \
            --format docker \
            -f "${dockerfile_path}" \
            -t "${image_name}" \
            "${context_path}" >> "$LOG_FILE" 2>&1 || {
                log_error "Failed to build image for ${service_name}. Check log for details: ${LOG_FILE}"
                exit 1
            }
        log_success "Image for ${service_name} built successfully."
    fi
}

# Starts a container, removing any old one with the same name first.
# Usage: start_container <service_name> <port_mapping> [env_vars...]
start_container() {
    local service_name=$1
    local port_mapping=$2
    shift 2
    local env_vars=("$@")
    local image_name="localhost/${PROJECT_NAME}/${service_name}:latest"
    local container_name="${PROJECT_NAME}-${service_name}-1"

    log_info "Starting container for service: ${service_name}..."

    # Stop and remove existing container to ensure a clean start
    if podman container exists "${container_name}"; then
        log_warn "Found existing container '${container_name}'. Removing it..."
        podman stop "${container_name}" >/dev/null || true
        podman rm "${container_name}" >/dev/null
    fi

    local podman_args=(
        "run"
        "-d"
        "--name=${container_name}"
        "--network=${NETWORK_NAME}"
        "--label=project=${PROJECT_NAME}"
    )

    if [[ -n "$port_mapping" ]]; then
        podman_args+=("-p" "${port_mapping}")
    fi

    # Add project-labeled volume for logs
    local abs_log_dir
    abs_log_dir=$(readlink -f "$LOG_DIR")
    podman_args+=("--volume=${abs_log_dir}:/app/logs:Z")
    podman_args+=("--label=project=${PROJECT_NAME}") # Label the container itself

    for env_var in "${env_vars[@]}"; do
        podman_args+=("-e" "${env_var}")
    done

    podman_args+=("${image_name}")

    if ! podman "${podman_args[@]}"; then
        log_error "Failed to start container '${container_name}'. Dumping logs..."
        podman logs "${container_name}" >> "$LOG_FILE" 2>&1 || log_warn "Could not retrieve logs for failed container."
        exit 1
    fi

    log_success "Container '${container_name}' started."
}

# Waits for a service's healthcheck to report 'healthy'.
# Usage: wait_for_service <service_name> [timeout_seconds]
wait_for_service() {
    local service_name=$1
    local timeout=${2:-120} # Default to 120 seconds
    local container_name="${PROJECT_NAME}-${service_name}-1"
    local end_time=$((SECONDS + timeout))

    log_info "Waiting for service '${service_name}' to become healthy (timeout: ${timeout}s)..."

    while [[ $SECONDS -lt $end_time ]]; do
        local health_status
        health_status=$(podman container inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' "${container_name}" 2>/dev/null || echo "error")

        case "$health_status" in
            "healthy")
                log_success "Service '${service_name}' is healthy."
                return 0
                ;;
            "no-healthcheck")
                log_warn "Service '${service_name}' has no healthcheck defined. Assuming it's ready."
                return 0
                ;;
            "unhealthy")
                log_error "Service '${service_name}' reported unhealthy status."
                podman logs "${container_name}" >> "$LOG_FILE" 2>&1
                exit 1
                ;;
            "starting")
                # Container is still starting up, healthcheck is running
                ;;
            "error")
                # Container might have crashed before healthcheck could run
                if ! podman container exists "${container_name}"; then
                    log_error "Container for '${service_name}' does not exist. It may have crashed."
                    exit 1
                fi
                ;;
        esac
        sleep 5
    done

    log_error "Timeout waiting for service '${service_name}' to become healthy."
    podman logs "${container_name}" >> "$LOG_FILE" 2>&1
    exit 1
}

# --- Script Main Execution ---
main() {
    setup_logging
    log_info "=== Starting Watchers Ecosystem Deployment ==="

    # --- 1. Network Setup ---
    if podman network exists "${NETWORK_NAME}"; then
        log_info "Network '${NETWORK_NAME}' already exists."
    else
        log_info "Creating network '${NETWORK_NAME}'..."
        podman network create "${NETWORK_NAME}"
        log_success "Network created."
    fi

    # --- 2. Image Builds ---
    log_info "--- Building service images (if needed) ---"
    build_if_needed "ecu" "./ecu/Dockerfile" "./ecu"
    build_if_needed "watchers_wave" "./watchers/watchers_tools/watchers_wave/Dockerfile" "./watchers/watchers_tools/watchers_wave"
    build_if_needed "watcher_focus" "./watchers/watchers_tools/watcher_focus/Dockerfile" "./watchers/watchers_tools/watcher_focus"
    build_if_needed "malla_watcher" "./watchers/watchers_tools/malla_watcher/Dockerfile" "./watchers/watchers_tools/malla_watcher"
    build_if_needed "harmony_controller" "./control/Dockerfile" "./control"
    build_if_needed "agent_ai" "./agent_ai/Dockerfile" "./agent_ai"

    # --- 3. Container Startup ---
    log_info "--- Starting service containers ---"

    # Services with no dependencies
    start_container "ecu" "8000:8000" "PYTHONPATH=/app" "PORT=8000"
    start_container "watchers_wave" "5000:5000" "PYTHONPATH=/app" "PORT=5000"
    start_container "watcher_focus" "6000:6000" "PYTHONPATH=/app" "PORT=6000"

    # Wait for them to be healthy
    wait_for_service "ecu"
    wait_for_service "watchers_wave"
    wait_for_service "watcher_focus"

    # Services that depend on the above
    start_container "malla_watcher" "5001:5001" "PYTHONPATH=/app" "MATRIZ_ECU_URL=http://ecu:8000" "TORUS_NUM_CAPAS=3" "TORUS_NUM_FILAS=4" "TORUS_NUM_COLUMNAS=5" "PORT=5001"
    wait_for_service "malla_watcher"

    # Harmony Controller depends on ECU and watchers
    HC_WATCHERS_CONFIG=$(cat <<-EOF
{
  "watchers_wave": "http://watchers_wave:5000",
  "watcher_focus": "http://watcher_focus:6000",
  "malla_watcher": "http://malla_watcher:5001"
}
EOF
)
    start_container "harmony_controller" "7000:7000" "PYTHONPATH=/app" "ECU_API_URL=http://ecu:8000/api/ecu" "WATCHERS_TOOLS_CONFIG=${HC_WATCHERS_CONFIG}" "PORT=7000"
    wait_for_service "harmony_controller"

    # Agent AI is the top-level service
    start_container "agent_ai" "9000:9000" "PYTHONPATH=/app" "HARMONY_CONTROLLER_URL=http://harmony_controller:7000" "AA_INITIAL_WAIT=10.0" "PORT=9000"
    # No need to wait for agent_ai, it's the final consumer

    log_success "=== Watchers Ecosystem Deployment Complete ==="
    log_info "Use 'podman ps -a --filter label=project=${PROJECT_NAME}' to see all project containers."
    log_info "Use 'podman logs -f ${PROJECT_NAME}-<service>-1' to follow logs."
}

# --- Run main function ---
main