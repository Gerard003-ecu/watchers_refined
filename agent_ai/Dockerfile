# Dockerfile para el servicio Agent AI
FROM python:3.12-slim

# Establecer variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear un usuario y grupo no-root 'appuser' con UID/GID fijos
RUN groupadd -r appgroup -g 1001 && \
    useradd -r -u 1001 -g appgroup appuser

# Copiar requirements.txt y instalar dependencias como root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . .

# Cambiar la propiedad de /app al usuario no-root
RUN chown -R appuser:appgroup /app

# Cambiar al usuario no-root para la ejecución
USER appuser

# Exponer el puerto específico de este servicio
EXPOSE 9000

# Comando para ejecutar el servicio como un paquete de Python.
# Esto establece 'agent_ai' como el paquete de nivel superior y ejecuta
# el módulo 'endpoints' que se encuentra en el sub-paquete 'api'.
# El WORKDIR es /app, y dentro de /app está el directorio agent_ai/,
# por lo que Python puede encontrar el paquete agent_ai.
CMD ["python", "-m", "agent_ai.api.endpoints"]
