#!/bin/bash
# ==============================================================================
# Script to Clean the Entire Watchers Ecosystem from Podman
#
# Features:
# - Robustness: Uses 'set -euo pipefail'.
# - Comprehensive Cleanup: Removes containers, images, network, and optionally volumes.
# - Efficiency: Uses filtering and bulk operations.
# - Clear Logging: Color-coded output.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -euo pipefail

# --- Configuration ---
PROJECT_NAME="watchers"
NETWORK_NAME="${PROJECT_NAME}_default"
CLEAN_VOLUMES=false

# --- Colors for Logging ---
COLOR_RESET='\033[0m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[0;33m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'

# --- Argument Parsing ---
if [[ "${1:-}" == "--with-volumes" ]]; then
    CLEAN_VOLUMES=true
fi

# --- Logging Functions ---
log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1"
}

log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $1"
}

log_warn() {
    echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $1"
}

# --- Script Main Execution ---
main() {
    log_info "=== Starting Full Cleanup of Watchers Ecosystem ==="

    # --- 1. Stop and Remove Containers ---
    log_info "Finding all containers for project '${PROJECT_NAME}'..."
    CONTAINER_IDS=$(podman ps -a --filter "label=project=${PROJECT_NAME}" --format "{{.ID}}")

    if [[ -z "$CONTAINER_IDS" ]]; then
        log_success "No containers found for project '${PROJECT_NAME}'."
    else
        log_info "Stopping and removing containers..."
        # Stop and then remove. The '|| true' handles cases where a container is already stopped.
        echo "$CONTAINER_IDS" | xargs --no-run-if-empty podman stop || true
        echo "$CONTAINER_IDS" | xargs --no-run-if-empty podman rm
        log_success "All project containers removed."
    fi

    # --- 2. Remove Images ---
    log_info "Finding all images for project '${PROJECT_NAME}'..."
    IMAGE_IDS=$(podman images --filter "reference=localhost/${PROJECT_NAME}/*" --format "{{.ID}}")

    if [[ -z "$IMAGE_IDS" ]]; then
        log_success "No images found for project '${PROJECT_NAME}'."
    else
        log_info "Removing images..."
        # Use --force to remove images even if they have multiple tags within the project
        echo "$IMAGE_IDS" | xargs --no-run-if-empty podman rmi --force
        log_success "All project images removed."
    fi

    # --- 3. Remove Network ---
    if podman network exists "${NETWORK_NAME}"; then
        log_info "Removing network '${NETWORK_NAME}'..."
        podman network rm "${NETWORK_NAME}"
        log_success "Network removed."
    else
        log_success "Network '${NETWORK_NAME}' not found."
    fi

    # --- 4. Remove Volumes (Optional) ---
    if [[ "$CLEAN_VOLUMES" == "true" ]]; then
        log_warn "--- Optional: Cleaning up project volumes ---"
        log_warn "This is a destructive action. The current implementation will prune ALL unused volumes on your system, not just for this project."
        log_warn "This is because Podman volumes are not easily associated with project labels."
        read -p "Are you sure you want to continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Pruning all unused volumes..."
            podman volume prune --force
            log_success "Volume cleanup complete."
        else
            log_info "Volume cleanup skipped by user."
        fi
    else
        log_info "Skipping volume cleanup. To remove all unused volumes, run with '--with-volumes'."
    fi


    log_success "=== Watchers Ecosystem Cleanup Complete ==="
}

# --- Run main function ---
main