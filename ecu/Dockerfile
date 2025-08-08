##### Dockerfile ecu #####
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Instalar curl y limpiar caché de apt
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip (opcional pero bueno)
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH="/app"

# Exponer el puerto del servicio
EXPOSE 8000

# Healthcheck (ahora curl debería existir)
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "matriz_ecu.py"]