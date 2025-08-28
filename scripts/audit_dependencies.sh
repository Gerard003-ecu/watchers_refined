#!/bin/bash

# --- Auditoría de Dependencias de Microservicios ---
# Este script realiza una auditoría de todos los archivos requirements.in en el proyecto.
# 1. Busca dependencias innecesarias (bloat) usando pipreqs.
# 2. Busca vulnerabilidades de seguridad usando pip-audit.

# --- Colores para el output ---
COLOR_ERROR='\033[0;31m'
COLOR_WARNING='\033[1;33m'
COLOR_INFO='\033[0;36m'
COLOR_SUCCESS='\033[0;32m'
COLOR_NC='\033[0m' # No Color

# --- Instalación de Herramientas ---
echo -e "${COLOR_INFO}Asegurando que las herramientas de auditoría estén instaladas...${COLOR_NC}"
# Instalar pipreqs y pip-audit usando uv pip install en el entorno virtual activo
# uv ya debe estar en el PATH del sistema
uv pip install --quiet pipreqs pip-audit

# --- Variables de Resumen ---
services_audited=0
bloat_warnings=0
security_errors=0
sync_errors=0

echo -e "\n${COLOR_INFO}--- Iniciando Auditoría de Dependencias ---${COLOR_NC}"

# --- Búsqueda de archivos requirements.in ---
# Excluir el directorio watchers_env, venv y el directorio requirements/
while IFS= read -r req_in_file; do
    ((services_audited++))
    service_dir=$(dirname "$req_in_file")
    # Limpiar el nombre del servicio para el informe
    service_name=$(echo "$service_dir" | sed 's|^\./||')
    req_txt_file="$service_dir/requirements.txt"

    echo -e "\n${COLOR_INFO}=======================================================================${COLOR_NC}"
    echo -e "${COLOR_INFO}Auditando Servicio: $service_name${COLOR_NC}"
    echo -e "${COLOR_INFO}=======================================================================${COLOR_NC}"

    # --- 1. Análisis de Bloat (Dependencias Innecesarias) ---
    echo -e "${COLOR_INFO}--- 1. Analizando dependencias innecesarias (Bloat)...${COLOR_NC}"

    # Generar un requirements.txt temporal con pipreqs
    # Usamos --force para sobrescribir si ya existe
    # pipreqs debe ejecutarse en el contexto del directorio del servicio
    (cd "$service_dir" && pipreqs --force --savepath "requirements.pipreqs.txt" . > /dev/null 2>&1)

    # Comparar requirements.in con el archivo generado por pipreqs
    # Leemos cada línea de requirements.in, extraemos el nombre del paquete y lo buscamos
    while IFS= read -r dep || [[ -n "$dep" ]]; do
        # Extraer el nombre del paquete (ignorando versiones, extras, etc.)
        pkg_name=$(echo "$dep" | sed -E 's/([a-zA-Z0-9\-_]+).*/\1/')

        # Ignorar líneas vacías, comentarios o líneas de restricciones (-c)
        if [[ -z "$pkg_name" || "$pkg_name" == \#* || "$pkg_name" == "-c" ]]; then
            continue
        fi

        if ! grep -q -i "^$pkg_name" "$service_dir/requirements.pipreqs.txt"; then
            echo -e "${COLOR_WARNING}[ADVERTENCIA] $service_name: La dependencia '$pkg_name' podría no ser utilizada.${COLOR_NC}"
            ((bloat_warnings++))
        fi
    done < "$req_in_file"

    # Limpiar el archivo temporal
    rm "$service_dir/requirements.pipreqs.txt"


    # --- 2. Análisis de Seguridad (Vulnerabilidades) ---
    echo -e "${COLOR_INFO}--- 2. Analizando vulnerabilidades de seguridad...${COLOR_NC}"

    # a. Verificar que requirements.txt exista y esté actualizado con uv pip compile --check
    if [ ! -f "$req_txt_file" ]; then
        echo -e "${COLOR_ERROR}[ERROR CRÍTICO] $service_name: No se encontró requirements.txt en $service_dir.${COLOR_NC}"
        ((sync_errors++))
        continue # Saltar al siguiente servicio
    fi

    # Usamos uv pip compile --check para verificar consistencia.
    # Se añade --constraint requirements/base.txt para que la verificación use las mismas restricciones que la compilación
    if ! uv pip compile --check "$req_in_file" --constraint requirements/base.txt > /dev/null 2>&1; then
        echo -e "${COLOR_ERROR}[ERROR CRÍTICO] $service_name: requirements.txt está desactualizado. Ejecute 'uv pip compile $req_in_file --constraint requirements/base.txt'.${COLOR_NC}"
        ((sync_errors++))
        continue # Saltar al siguiente servicio
    fi

    # b. Ejecutar pip-audit
    # Capturamos la salida para poder mostrarla solo si hay vulnerabilidades.
    audit_output=$(pip-audit -r "$req_txt_file")

    if [ $? -ne 0 ]; then
        echo -e "${COLOR_ERROR}[ERROR CRÍTICO] $service_name: Se encontraron vulnerabilidades de seguridad:${COLOR_NC}"
        echo "$audit_output"
        ((security_errors++))
    else
        echo -e "${COLOR_SUCCESS}[OK] $service_name: No se encontraron vulnerabilidades.${COLOR_NC}"
    fi

done < <(find . -type f -name "requirements.in" -not -path "./watchers_env/*" -not -path "./venv/*" -not -path "./requirements/*")

# --- Informe Final ---
echo -e "\n${COLOR_INFO}=======================================================================${COLOR_NC}"
echo -e "${COLOR_INFO}--- Auditoría Finalizada: Resumen ---${COLOR_NC}"
echo -e "${COLOR_INFO}=======================================================================${COLOR_NC}"
echo -e "Servicios auditados: $services_audited"
echo -e "${COLOR_WARNING}Advertencias de Bloat: $bloat_warnings${COLOR_NC}"
echo -e "${COLOR_ERROR}Errores de Sincronización (req.txt): $sync_errors${COLOR_NC}"
echo -e "${COLOR_ERROR}Errores Críticos de Seguridad: $security_errors${COLOR_NC}"

# Salir con un código de error si se encontraron problemas críticos
if [ $security_errors -gt 0 ] || [ $sync_errors -gt 0 ]; then
    exit 1
fi

exit 0
