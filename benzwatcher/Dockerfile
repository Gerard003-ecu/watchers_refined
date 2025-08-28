FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app/benzwatcher

# Instalar uv
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    apt-get remove -y curl && apt-get clean

# Añadir uv al PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Copiar requirements.txt y luego instalar dependencias
COPY benzwatcher/requirements.txt .
RUN uv pip sync requirements.txt

# Copiar el resto de los archivos
COPY . .

# Exponer el puerto en el que FastAPI correrá (en este ejemplo 8000)
EXPOSE 8000

# Comando de inicio: ejecutar la API de BenzWatcher
CMD ["uvicorn", "api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]
