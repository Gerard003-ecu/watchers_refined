#!/bin/bash

# --- Verificación de Consistencia de Dependencias del Ecosistema Watchers ---

# --- Configuración de Rutas ---
# Obtener la ruta absoluta del directorio raíz del proyecto
PROJECT_ROOT=$(pwd)
# Obtener la ruta absoluta del archivo base.txt
BASE_REQUIREMENTS_TXT="$PROJECT_ROOT/requirements/base.txt"

# Contadores para el resumen
consistent_services=0
inconsistent_services=0

echo "--- Verificando Consistencia de Dependencias del Ecosistema Watchers ---"
echo ""

# Usar process substitution para evitar problemas de subshell con los contadores
while IFS= read -r req_in_file; do
    service_dir=$(dirname "$req_in_file")
    # Limpiar el nombre del servicio para el informe
    service_name=$(echo "$service_dir" | sed 's|^\./||')
    req_txt_file="$service_dir/requirements.txt"
    tmp_output_file="${req_txt_file}.tmp"

    # 1. Verificar si requirements.txt existe
    if [ ! -f "$req_txt_file" ]; then
        echo "[ ! ] $service_name: ERROR - No se encontró requirements.txt en $service_dir."
        ((inconsistent_services++))
        continue
    fi

    # 2. Compilar a un archivo temporal y comparar con el existente
    uv pip compile "$req_in_file" --constraint "$BASE_REQUIREMENTS_TXT" -o "$tmp_output_file" >/dev/null 2>&1

    # 3. Comparar los archivos, ignorando TODAS las líneas de comentario para una comparación robusta.
    # `grep -v "^\s*#"` filtra todas las líneas que comienzan con #, con o sin espacios en blanco.
    if ! diff -q <(grep -v "^\s*#" "$req_txt_file") <(grep -v "^\s*#" "$tmp_output_file") >/dev/null 2>&1; then
        echo "[ ! ] $service_name: ERROR - requirements.txt está desactualizado. Por favor, ejecute './scripts/compile_requirements.sh'."
        ((inconsistent_services++))
    else
        echo "[ ✓ ] $service_name: OK"
        ((consistent_services++))
    fi

    # Limpiar el archivo temporal
    rm "$tmp_output_file"

done < <(find . -type f -name "requirements.in" -not -path "./watchers_env/*" -not -path "./requirements/*")

echo ""
echo "--- Resumen ---"
echo "$consistent_services servicios consistentes."
echo "$inconsistent_services servicios requieren atención."

# Salir con un código de error si hay servicios inconsistentes
if [ "$inconsistent_services" -gt 0 ]; then
    exit 1
fi
