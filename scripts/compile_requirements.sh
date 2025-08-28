#!/bin/bash
set -e

# --- Ejecución Principal ---
echo "--- Compilando Archivos de Requisitos del Monorepo con uv ---"

# 1. Compilar los requisitos base y de desarrollo en la raíz
echo "Compilando requisitos raíz (base y dev)..."
uv pip compile requirements.in -o requirements.txt
uv pip compile requirements-dev.in -o requirements-dev.txt

# 2. Descubrir y compilar los requisitos de cada servicio
echo "Compilando requisitos de los servicios..."
find . -path ./requirements.in -prune -o \
       -path ./watchers_env -prune -o \
       -name "requirements.in" -print | while read -r req_in; do

    dir=$(dirname "$req_in")
    req_txt="${req_in%.in}.txt"

    echo "Compilando $req_in -> $req_txt"

    # Compilar, usando los requisitos base de la raíz como restricciones
    uv pip compile "$req_in" -o "$req_txt" --constraint requirements.txt
done

echo "--- Compilación Completa con uv ---"
