# 1. Imagen Base: Usa una imagen base de Python específica y slim.
FROM python:3.12-slim-bookworm

# 2. Entorno: Establece WORKDIR y variables de entorno.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Instalar uv
RUN apt-get update && apt-get install -y curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && curl -LsSf https://astral.sh/uv/install.sh | sh

# Añadir uv al PATH
ENV PATH="/root/.cargo/bin:$PATH"

# 3. Optimización de Caché de Dependencias:
COPY optical_controller/requirements.txt .
RUN uv pip sync requirements.txt

# Copia el resto del código de la aplicación.
COPY . .

# 4. Seguridad: Crea un usuario no-root y cambia a él.
RUN useradd -m app
USER app

# 5. Metadatos y Ejecución:
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8001/api/health || exit 1

CMD ["python", "optical_controller.py"]
