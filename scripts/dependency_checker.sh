#!/bin/bash

# --- Verificación de Consistencia de Dependencias del Ecosistema Watchers ---

# Contadores para el resumen
consistent_services=0
inconsistent_services=0

echo "--- Verificando Consistencia de Dependencias del Ecosistema Watchers ---"
echo ""

# Usar process substitution para evitar problemas de subshell con los contadores
# Excluir el directorio watchers_env y el directorio requirements/
while IFS= read -r req_in_file; do
    service_dir=$(dirname "$req_in_file")
    # Limpiar el nombre del servicio para el informe
    service_name=$(echo "$service_dir" | sed 's|^\./||')
    req_txt_file="$service_dir/requirements.txt"

    # 1. Verificar si requirements.txt existe
    if [ ! -f "$req_txt_file" ]; then
        echo "[ ! ] $service_name: ERROR - No se encontró requirements.txt en $service_dir."
        ((inconsistent_services++))
        continue
    fi

    # 2. Usar uv pip compile --check para verificar la consistencia
    # Se redirige la salida a /dev/null para no ensuciar el informe
    # Se añade --constraint requirements/base.txt para que la verificación use las mismas restricciones que la compilación
    if ! uv pip compile --check --quiet "$req_in_file" --constraint requirements/base.txt >/dev/null 2>&1; then
        echo "[ ! ] $service_name: ERROR - requirements.txt está desactualizado. Por favor, ejecute 'uv pip compile $req_in_file --constraint requirements/base.txt'."
        ((inconsistent_services++))
    else
        echo "[ ✓ ] $service_name: OK"
        ((consistent_services++))
    fi
done < <(find . -type f -name "requirements.in" -not -path "./watchers_env/*" -not -path "./requirements/*")

echo ""
echo "--- Resumen ---"
echo "$consistent_services servicios consistentes."
echo "$inconsistent_services servicios requieren atención."
