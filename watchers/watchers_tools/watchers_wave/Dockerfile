##### Dockerfile watchers_wave #####
FROM python:3.12-slim

# Configuración base
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Instalar dependencias del sistema (si son necesarias)
# RUN apt-get update && apt-get install -y <paquetes> && rm -rf /var/lib/apt/lists/*

# Actualizar herramientas de Python
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias primero para aprovechar la caché
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . /app

# Configuración final
ENV PYTHONPATH="/app"

# Exponer el puerto del servicio
EXPOSE 5000

# Healthcheck para verificar si el servicio Flask responde
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# Comando de inicio
CMD ["python", "watchers_wave.py"]