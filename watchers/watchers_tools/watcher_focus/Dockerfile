##### Dockerfile watcher_focus #####
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# --- PASO 1: Instalar dependencias del sistema (curl) ---
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# --- PASO 2: Actualizar herramientas de Python ---
RUN pip install --upgrade pip setuptools wheel

# --- PASO 3: Instalar dependencias de Python ---
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- PASO 4: Copiar código y configurar ---
# --- PASO CORREGIDO Y MÁS EXPLÍCITO: Crear estructura y copiar código ---
# 1. Crear la estructura de directorios esperada para el paquete dentro de /app
#    El path dentro del contenedor debería ser /app/watchers/watchers_tools/watcher_focus
RUN mkdir -p /app/watchers/watchers_tools/watcher_focus

# 2. Copiar el contenido del directorio del servicio (el contexto '.')
#    dentro de la estructura de carpetas creada.
#    Desde el contexto (./watchers/watchers_tools/watcher_focus), copiamos todo '.'
#    a /app/watchers/watchers_tools/watcher_focus/
COPY . /app/watchers/watchers_tools/watcher_focus
ENV PYTHONPATH="/app"
EXPOSE 6000

# --- PASO 5: Healthcheck ---
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:6000/api/health || exit 1

# --- PASO 6: Comando de inicio ---
CMD ["python", "/app/watchers/watchers_tools/watcher_focus/watcher_focus.py"]