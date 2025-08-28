FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar uv
RUN apt-get update && apt-get install -y curl \
    libgl1-mesa-glx libglib2.0-0 \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

# Añadir uv al PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Copiar el archivo unificado de requerimientos para aprovechar la caché de Docker
COPY requirements-dev.txt .

# Instalar las dependencias
RUN uv pip sync requirements-dev.txt

# Copiar el resto del código fuente
COPY . .

# Comando para ejecutar los tests de integración
CMD ["pytest", "tests/integration_tests.py"]
