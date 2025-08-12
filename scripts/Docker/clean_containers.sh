#!/bin/bash
# ==============================================================================
# Script to Clean the Entire Watchers Ecosystem using Docker Compose
#
# This script stops and removes all containers, networks, and optionally
# volumes defined in the docker-compose.yml file.
# ==============================================================================

# --- Strict Mode ---
set -e

# --- Colors for Logging ---
COLOR_RESET='\033[0m'
COLOR_GREEN='\033[0;32m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'
COLOR_YELLOW='\033[0;33m'

# --- Argument Parsing ---
# Check if the first argument is '--with-volumes'
CLEAN_VOLUMES=false
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

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $1"
}

log_warn() {
    echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $1"
}

# --- Main Execution ---
main() {
    log_info "=== Starting Full Cleanup of Watchers Ecosystem ==="

    # --- 1. Check for Docker Compose ---
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose could not be found."
        log_error "Please ensure Docker Desktop is installed and running."
        exit 1
    fi
    log_info "docker-compose found."

    # --- 2. Clean Environment ---
    if [[ "$CLEAN_VOLUMES" == "true" ]]; then
        log_warn "--- Cleaning up containers, networks, AND volumes! ---"
        # The 'down' command stops containers and removes containers, networks, and volumes.
        docker-compose down --volumes
    else
        log_info "--- Cleaning up containers and networks ---"
        # By default, 'down' only removes containers and networks.
        docker-compose down
        log_info "To remove volumes as well, run this script with the '--with-volumes' flag."
    fi

    log_success "=== Ecosystem cleanup complete ==="
}

# --- Run main function ---
main
