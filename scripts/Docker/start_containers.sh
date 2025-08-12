#!/bin/bash
# ==============================================================================
# Script to Start the Watchers Ecosystem using Docker Compose
#
# This script is a wrapper for 'docker-compose' and uses the main
# 'docker-compose.yml' file to start the entire application stack.
# ==============================================================================

# --- Strict Mode ---
# Exit immediately if a command exits with a non-zero status.
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
    log_info "=== Starting Watchers Ecosystem with Docker Compose ==="

    # --- 1. Check for Docker Compose ---
    if ! command -v docker-compose &> /dev/null; then
        log_error "docker-compose could not be found."
        log_error "Please ensure Docker Desktop is installed and running."
        exit 1
    fi
    log_info "docker-compose found."

    # --- 2. Check for .env file ---
    # The docker-compose.yml uses variables that should be in a .env file.
    if [ ! -f ".env" ]; then
        log_error "'.env' file not found in the root directory."
        log_error "This project requires a .env file to define environment-specific variables like ports."
        log_error "Please create one (e.g., from an '.env.example' if available) before running this script."
        exit 1
    fi
    log_info "'.env' file found."

    # --- 3. Start Containers ---
    log_info "Building and starting all services in detached mode..."
    log_info "This may take a while on the first run as images are built."

    # The --build flag rebuilds images if the Dockerfile or context has changed.
    # The -d flag runs containers in the background (detached mode).
    docker-compose up --build -d

    log_success "Docker Compose has started the services."

    # --- 4. Show Status ---
    log_info "Current status of all project containers:"
    docker-compose ps

    log_success "=== Ecosystem started successfully ==="
    log_info "Use 'scripts/stop_containers.sh' to stop the services."
    log_info "Use 'docker-compose logs -f <service_name>' to view logs."
}

# --- Run main function ---
main
