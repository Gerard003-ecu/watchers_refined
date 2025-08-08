##### Dockerfile harmony_controller (control) #####
FROM python:3.12-slim

# Dependencias del sistema (ej. para GLib que podría necesitar alguna librería)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV PYTHONPATH="/app"

# Exponer el puerto del servicio
EXPOSE 7000

# Healthcheck para verificar si el servicio Flask responde (usando /api/harmony/state como ejemplo si no hay /api/health)
# Ajusta la URL si tienes un endpoint de salud específico
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:7000/api/health || exit 1

  CMD ["python", "-m", "harmony_controller"]