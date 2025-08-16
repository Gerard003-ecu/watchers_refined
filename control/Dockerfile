# Dockerfile para el servicio Harmony Controller
FROM python:3.12-slim

# Establecer variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
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
EXPOSE 7000

# Comando para ejecutar el servicio como un paquete de Python.
# Esto ejecuta el fichero __main__.py dentro del directorio/paquete 'control'.
CMD ["python", "-m", "control"]
