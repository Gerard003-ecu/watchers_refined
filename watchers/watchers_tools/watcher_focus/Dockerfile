# Dockerfile para el servicio Watcher Focus
FROM python:3.12-slim

# Establecer variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# El WORKDIR es /app, que se añade a PYTHONPATH para que los módulos sean importables
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Instalar uv
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

# Añadir uv al PATH
ENV PATH="/root/.cargo/bin:$PATH"

# Crear un usuario y grupo no-root 'appuser' con UID/GID fijos
RUN groupadd -r appgroup -g 1001 && \
    useradd -r -u 1001 -g appgroup appuser

# Copiar el fichero de requisitos específico del servicio y instalar dependencias
COPY watchers/watchers_tools/watcher_focus/requirements.txt .
RUN uv pip sync requirements.txt

# Copiar los directorios de los paquetes necesarios desde el contexto de build (la raíz del proyecto)
COPY watchers/ /app/watchers/
COPY common/ /app/common/

# Cambiar la propiedad de /app al usuario no-root
RUN chown -R appuser:appgroup /app

# Cambiar al usuario no-root para la ejecución
USER appuser

# Exponer el puerto específico de este servicio
EXPOSE 6000

# Ejecutar el servicio como un módulo anidado dentro del paquete 'watchers'
CMD ["python", "-m", "watchers.watchers_tools.watcher_focus.watcher_focus"]
