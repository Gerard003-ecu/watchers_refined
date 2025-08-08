##### Dockerfile malla_watcher #####
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# --- PASO 1: Instalar dependencias del sistema (curl) ---
  RUN apt-get update && apt-get install -y --no-install-recommends curl \
  && rm -rf /var/lib/apt/lists/*

# --- PASO 2: Actualizar herramientas de Python ---
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- PASO CORREGIDO Y MÁS EXPLÍCITO: Crear estructura y copiar código ---
# 1. Crear la estructura de directorios esperada para el paquete dentro de /app
#    El path dentro del contenedor debería ser /app/watchers/watchers_tools/malla_watcher
RUN mkdir -p /app/watchers/watchers_tools/malla_watcher

# 2. Copiar el contenido del directorio del servicio (el contexto '.')
#    dentro de la estructura de carpetas creada.
#    Desde el contexto (./watchers/watchers_tools/malla_watcher), copiamos todo '.'
#    a /app/watchers/watchers_tools/malla_watcher/
COPY . /app/watchers/watchers_tools/malla_watcher

# Configuración final
ENV PYTHONPATH="/app"

# Exponer el puerto del servicio
EXPOSE 5001

# Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5001/api/health || exit 1

# Comando de inicio (Asegúrate de que el path es correcto dentro del contenedor)
# Ahora el script está en /app/watchers/watchers_tools/malla_watcher/malla_watcher.py
CMD ["python", "/app/watchers/watchers_tools/malla_watcher/malla_watcher.py"]