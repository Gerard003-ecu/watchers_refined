# config_agent/Dockerfile
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar uv
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get remove -y curl && apt-get clean

# Añadir uv al PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Copiar el código del agente y sus dependencias
COPY config_agent/requirements.txt .
RUN uv pip sync requirements.txt

# Copiar el código del agente y el módulo común
COPY config_agent/ /app/config_agent/
COPY common/ /app/common/

# Copiar los archivos de configuración que necesita analizar
COPY docker-compose.yml ./
COPY config/ecosystem_topology.yml ./config/

# El comando que se ejecuta cuando el contenedor arranca
CMD ["python", "-m", "config_agent.config_agent"]