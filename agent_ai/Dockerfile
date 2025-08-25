# Dockerfile para el servicio Agent AI
FROM python:3.12-slim

# Establecer variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# El WORKDIR es /app, que se añade a PYTHONPATH para que los módulos sean importables
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Crear un usuario y grupo no-root
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copiar el fichero de requisitos específico del servicio y instalar dependencias
COPY requirements/agent_ai.txt .
RUN pip install --no-cache-dir -r agent_ai.txt

# Copiar los directorios de los paquetes necesarios desde el contexto de build (la raíz del proyecto)
COPY agent_ai/ /app/agent_ai/
COPY common/ /app/common/

# Cambiar la propiedad de /app al usuario no-root
RUN chown -R appuser:appgroup /app

# Exponer el puerto del servicio
EXPOSE 9000

# Cambiar al usuario no-root
USER appuser

# Comando para ejecutar el servicio como un paquete de Python.
# Esto establece 'agent_ai' como el paquete de nivel superior y ejecuta
# el módulo 'endpoints' que se encuentra en el sub-paquete 'api'.
# El WORKDIR es /app, y dentro de /app está el directorio agent_ai/,
# por lo que Python puede encontrar el paquete agent_ai.
CMD ["python", "-m", "agent_ai.api.endpoints"]
