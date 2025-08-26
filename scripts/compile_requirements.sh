#!/bin/bash
set -e

# --- Función de Bootstrap ---
# Asegura que pip-tools esté instalado para poder ejecutar el script.
ensure_pip_tools() {
    if ! command -v pip-compile &> /dev/null; then
        echo "pip-compile no encontrado. Instalando pip-tools..."
        python3 -m pip install pip-tools
    fi
}

# --- Ejecución Principal ---
echo "--- Compilando Archivos de Requisitos del Monorepo ---"

# 1. Asegurar que las herramientas necesarias existan
ensure_pip_tools

# 2. Compilar los requisitos base y de desarrollo en la raíz
echo "Compilando requisitos raíz (base y dev)..."
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt

# 3. Descubrir y compilar los requisitos de cada servicio
echo "Compilando requisitos de los servicios..."
find . -path ./requirements.in -prune -o \
       -path ./watchers_env -prune -o \
       -name "requirements.in" -print | while read -r req_in; do

    dir=$(dirname "$req_in")
    req_txt="${req_in%.in}.txt"

    echo "Compilando $req_in -> $req_txt"

    # Compilar, usando los requisitos base de la raíz como restricciones
    pip-compile "$req_in" -o "$req_txt" -c requirements.txt
done

echo "--- Compilación Completa ---"
