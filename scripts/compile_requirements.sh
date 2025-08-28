#!/bin/bash
set -e

# --- Configuración de Rutas ---
# Obtener la ruta absoluta del directorio raíz del proyecto
PROJECT_ROOT=$(pwd)
# Obtener la ruta absoluta del archivo base.txt
BASE_REQUIREMENTS_TXT="$PROJECT_ROOT/requirements/base.txt"

# --- Ejecución Principal ---
echo "--- Compilando Archivos de Requisitos del Monorepo con uv ---"

# 1. Compilar los requisitos base y de desarrollo en la raíz
echo "Compilando requisitos raíz (base y dev)..."
# Asegurarse de que requirements/base.txt se compile primero sin restricciones
uv pip compile requirements/base.in -o "$BASE_REQUIREMENTS_TXT"
uv pip compile requirements/dev.in -o requirements/dev.txt --constraint "$BASE_REQUIREMENTS_TXT"

# 2. Descubrir y compilar los requisitos de cada servicio
echo "Compilando requisitos de los servicios..."
# Excluir el directorio watchers_env y el directorio requirements/
find . -type f -name "requirements.in" -not -path "./watchers_env/*" -not -path "./requirements/*" | while read -r req_in; do

    dir=$(dirname "$req_in")
    req_txt="${req_in%.in}.txt"

    echo "Compilando $req_in -> $req_txt"

    # Compilar, usando los requisitos base de la raíz como restricciones
    # Usar la ruta absoluta para la restricción
    uv pip compile "$req_in" -o "$req_txt" --constraint "$BASE_REQUIREMENTS_TXT"
done

echo "--- Compilación Completa con uv ---"
