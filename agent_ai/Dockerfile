##### Dockerfile agent_ai #####
FROM python:3.12-slim

WORKDIR /app

# Instalar compiladores para dependencias nativas (si son necesarias)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Añadido para coherencia y posibles imports
ENV PYTHONPATH="/app"

# Exponer el puerto del servicio
EXPOSE 9000

# Healthcheck para verificar si el servicio Flask responde
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:9000/api/health || exit 1

# CMD corregido para usar ruta explícita
CMD ["python", "-m", "api.endpoints"]