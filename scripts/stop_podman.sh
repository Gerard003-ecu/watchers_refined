#!/bin/bash
# ==============================================================================
# Script to Stop the Watchers Ecosystem Containers
#
# Features:
# - Robustness: Uses 'set -euo pipefail'.
# - Efficiency: Stops all project containers with a single command.
# - Clear Logging: Color-coded output for better readability.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -euo pipefail

# --- Configuration ---
PROJECT_NAME="watchers"

# --- Colors for Logging ---
COLOR_RESET='\033[0m'
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'

# --- Logging Functions ---
log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1"
}

log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $1"
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $1"
}

# --- Script Main Execution ---
main() {
    log_info "=== Stopping Watchers Ecosystem ==="

    log_info "Finding running containers for project '${PROJECT_NAME}'..."
    # Find containers by the project label we added in the start script
    CONTAINER_IDS=$(podman ps --filter "label=project=${PROJECT_NAME}" --format "{{.ID}}")

    if [[ -z "$CONTAINER_IDS" ]]; then
        log_success "No running containers found for project '${PROJECT_NAME}'. Nothing to do."
        exit 0
    fi

    log_info "Found running containers. Stopping them now..."
    echo "$CONTAINER_IDS" | xargs --no-run-if-empty podman stop

    log_success "All containers for project '${PROJECT_NAME}' have been stopped."
}

# --- Run main function ---
main