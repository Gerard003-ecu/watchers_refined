#!/bin/bash
# ==============================================================================
# Script to Orchestrate the Watchers Ecosystem with podman-compose
#
# Description:
# This script acts as a robust orchestrator for the "watchers" microservices.
# It leverages podman-compose to manage the application lifecycle, using the
# docker-compose.yml file as the single source of truth for all service
# configurations (ports, environment variables, dependencies, healthchecks).
#
# Features:
# - Orchestrator, not Implementor: Delegates all container build, run, and
#   networking logic to podman-compose.
# - Strict & Safe: Uses 'set -euo pipefail' for robust error handling.
# - Clean State: Ensures a clean environment by tearing down previous
#   deployments before starting.
# - Rich Logging: Provides color-coded logs and saves a detailed log file
#   for each run.
# - Health-aware: Verifies that all services are healthy after startup.
# ==============================================================================

# --- Strict Mode & Error Handling ---
set -euo pipefail

# --- Configuration ---
# The project name is now read from the .env file by podman-compose.
# We only need log configuration here.
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

# --- Utility Functions ---

# Creates the log directory if it doesn't exist.
setup_logging() {
    mkdir -p "$LOG_DIR"
    # Create a new log file for this run
    >"$LOG_FILE"
    log_info "Logging to ${LOG_FILE}"
}

# --- Script Main Execution ---
main() {
    setup_logging
    log_info "=== Starting Watchers Ecosystem Deployment via podman-compose ==="

    # 1. Clean Up Previous Environment
    # Stops and removes containers, networks, and volumes defined in the compose file.
    # --remove-orphans removes containers for services not defined in the compose file.
    log_info "Stopping and removing any existing project containers and networks..."
    podman-compose down --remove-orphans >> "$LOG_FILE" 2>&1 || log_warn "No existing containers to remove, or an error occurred during cleanup. Continuing..."

    # 2. Prepare Host System
    log_info "Ensuring log directory exists at '${LOG_DIR}'..."
    # The mkdir -p in setup_logging handles this. This log is for clarity.

    # 3. Build All Service Images
    # podman-compose build is idempotent. It will only rebuild if the context has changed.
    log_info "Building all service images..."
    if ! podman-compose build >> "$LOG_FILE" 2>&1; then
        log_error "Image build failed. Check the log file for details: ${LOG_FILE}"
        exit 1
    fi
    log_info "Image build process complete."


    # 4. Start the Entire Ecosystem
    # podman-compose up -d starts all services in the background.
    # It respects the `depends_on` and `healthcheck` directives in the compose file.
    log_info "Starting all services in detached mode..."
    if ! podman-compose up -d >> "$LOG_FILE" 2>&1; then
        log_error "Failed to start services. Check the log file for details: ${LOG_FILE}"
        # Dump logs from all services for easier debugging
        podman-compose logs >> "$LOG_FILE" 2>&1
        exit 1
    fi
    log_success "All services have been started."

    # 5. Final Health Verification
    # Give services a moment to stabilize and for healthchecks to report their status.
    log_info "Waiting for services to stabilize (15s)..."
    sleep 15

    log_info "Final status of all project containers:"
    podman-compose ps | tee -a "$LOG_FILE"

    # Check if any container is 'unhealthy'.
    # The `ps` command output includes the health status.
    if podman-compose ps | grep -q 'unhealthy'; then
        log_error "One or more services are unhealthy. Dumping all logs for inspection..."
        podman-compose logs > "${LOG_DIR}/podman_compose_error_unhealthy.log" 2>&1
        log_error "Deployment failed. See full logs in ${LOG_DIR}/podman_compose_error_unhealthy.log and ${LOG_FILE}"
        exit 1
    fi

    log_success "=== Watchers Ecosystem Deployment Complete. All services are running and healthy. ==="
}

# --- Run main function ---
main