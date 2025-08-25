#!/bin/bash
set -e
echo "--- Compilando Archivos de Requisitos del Monorepo ---"

# 1. Compilar los requisitos base y de desarrollo en la raíz
echo "Compilando requisitos raíz (base y dev)..."
pip-compile requirements.in -o requirements.txt
pip-compile requirements-dev.in -o requirements-dev.txt

# 2. Descubrir y compilar los requisitos de cada servicio
# Usamos find para buscar todos los requirements.in, excluyendo la raíz y el venv
find . -path ./requirements.in -prune -o \
       -path ./watchers_env -prune -o \
       -name "requirements.in" -print | while read -r req_in; do

    # Obtener el directorio y el archivo .txt correspondiente
    dir=$(dirname "$req_in")
    req_txt="${req_in%.in}.txt"

    echo "Compilando $req_in -> $req_txt"

    # Compilar, usando los requisitos base de la raíz como restricciones
    pip-compile "$req_in" -o "$req_txt" -c requirements.txt
done

echo "--- Compilación Completa ---"
