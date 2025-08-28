# 1. Imagen Base: Usa una imagen base de Python específica y slim.
FROM python:3.12-slim-bookworm

# 2. Entorno: Establece WORKDIR y variables de entorno.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Instalar dependencias del sistema (curl para HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Optimización de Caché de Dependencias:
COPY watchers/watchers_tools/solenoid_watcher/controller/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación.
COPY . .

# 4. Seguridad: Crea un usuario no-root y cambia a él.
RUN useradd -m app
USER app

# 5. Metadatos y Ejecución:
EXPOSE 5002
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5002/api/health || exit 1

CMD ["python", "solenoid_controller.py"]
