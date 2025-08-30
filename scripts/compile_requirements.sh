#!/bin/bash
set -e

# --- Configuración de Rutas ---
# Obtener la ruta absoluta del directorio raíz del proyecto
PROJECT_ROOT=$(pwd)
# Obtener la ruta absoluta del archivo base.txt
BASE_REQUIREMENTS_TXT="$PROJECT_ROOT/requirements/base.txt"

# --- Ejecución Principal ---
echo "--- Compilando Archivos de Requisitos del Monorepo con uv ---"

# 2. Descubrir y compilar los requisitos de cada servicio
echo "Compilando requisitos de los servicios..."
# Excluir el directorio watchers_env y el directorio requirements/
while IFS= read -r req_in; do
    dir=$(dirname "$req_in")
    req_txt="${req_in%.in}.txt"

    echo "Compilando $req_in -> $req_txt"
    # Compilar, usando los requisitos base de la raíz como restricciones
    # La restricción ahora se lee desde el propio archivo .in
    uv pip compile "$req_in" -o "$req_txt"
done < <(find . -type f -name "requirements.in" -not -path "./watchers_env/*" -not -path "./requirements/*")

echo "--- Compilación Completa con uv ---"
