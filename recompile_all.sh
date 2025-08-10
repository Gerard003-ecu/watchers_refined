#!/bin/bash
echo "--- Re-compilando todos los archivos requirements.txt para el entorno actual ---"
# Encuentra todos los .in y ejecuta pip-compile en ellos
find . -type f -name "requirements.in" -not -path "./watchers_env/*" | while read req_in_file; do
    service_dir=$(dirname "$req_in_file")
    echo "Compilando: $req_in_file"
    pip-compile "$req_in_file" --output-file="$service_dir/requirements.txt"
done
echo "--- Re-compilación completada ---"