#!/bin/bash
# ==============================================================================
# Script to Stop the Watchers Ecosystem using Docker Compose
# ==============================================================================

# --- Strict Mode ---
set -e

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

# --- Main Execution ---
main() {
    log_info "=== Stopping Watchers Ecosystem Containers ==="

    # --- 1. Check for Docker Compose ---
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose could not be found."
        log_error "Please ensure Docker Desktop is installed and running."
        exit 1
    fi
    log_info "docker-compose found."

    # --- 2. Stop Containers ---
    log_info "Stopping all services... This may take a moment."
    docker-compose stop

    log_success "=== Ecosystem stopped successfully ==="
    log_info "Use 'scripts/start_containers.sh' to start them again."
    log_info "Use 'scripts/clean_containers.sh' to remove them completely."
}

# --- Run main function ---
main
