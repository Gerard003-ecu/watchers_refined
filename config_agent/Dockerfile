# config_agent/Dockerfile
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el código del agente y sus dependencias
COPY requirements/config_agent.txt .
RUN pip install --no-cache-dir -r config_agent.txt

# Copiar el código del agente y el módulo común
COPY config_agent/ /app/config_agent/
COPY common/ /app/common/

# Copiar los archivos de configuración que necesita analizar
COPY docker-compose.yml ./
COPY config/ecosystem_topology.yml ./config/

# El comando que se ejecuta cuando el contenedor arranca
CMD ["python", "-m", "config_agent.config_agent"]