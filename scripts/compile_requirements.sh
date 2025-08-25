#!/bin/bash
set -e
echo "--- Compilando Archivos de Requisitos ---"

# Compilar la base primero
pip-compile requirements/base.in -o requirements/base.txt

# Compilar el resto, restringiendo a las versiones de la base
pip-compile requirements/dev.in -o requirements/dev.txt
pip-compile requirements/watchers_tools_watchers_wave.in -o requirements/watchers_tools_watchers_wave.txt
pip-compile requirements/watchers_tools_solenoid_watcher_controller.in -o requirements/watchers_tools_solenoid_watcher_controller.txt
pip-compile requirements/watchers_tools_solenoid_watcher_model.in -o requirements/watchers_tools_solenoid_watcher_model.txt
pip-compile requirements/watchers_tools_watcher_focus.in -o requirements/watchers_tools_watcher_focus.txt
pip-compile requirements/watchers_tools_malla_watcher.in -o requirements/watchers_tools_malla_watcher.txt
pip-compile requirements/agent_ai.in -o requirements/agent_ai.txt
pip-compile requirements/benzwatcher.in -o requirements/benzwatcher.txt
pip-compile requirements/config_agent.in -o requirements/config_agent.txt
pip-compile requirements/ecu.in -o requirements/ecu.txt
pip-compile requirements/dashboard.in -o requirements/dashboard.txt
pip-compile requirements/atomic_piston.in -o requirements/atomic_piston.txt
pip-compile requirements/control.in -o requirements/control.txt
pip-compile requirements/optical_controller.in -o requirements/optical_controller.txt
pip-compile requirements/cogniboard.in -o requirements/cogniboard.txt
pip-compile requirements/watcher_security.in -o requirements/watcher_security.txt

echo "--- Compilación Completa ---"
