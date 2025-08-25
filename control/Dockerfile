# Dockerfile para el servicio Harmony Controller
FROM python:3.12-slim

# Establecer variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# El WORKDIR es /app, que se añade a PYTHONPATH para que los módulos sean importables
WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear un usuario y grupo no-root
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copiar el fichero de requisitos específico del servicio y instalar dependencias
COPY requirements/control.txt .
RUN pip install --no-cache-dir -r control.txt

# Copiar los directorios de los paquetes necesarios desde el contexto de build (la raíz del proyecto)
# Esto asegura que la estructura de paquetes se mantiene dentro del contenedor
COPY control/ /app/control/
COPY common/ /app/common/

# Cambiar la propiedad de /app al usuario no-root
RUN chown -R appuser:appgroup /app

# Exponer el puerto del servicio
EXPOSE 7000

# Cambiar al usuario no-root
USER appuser

# Comando para ejecutar el servicio como un paquete de Python.
# Esto ejecuta el fichero __main__.py dentro del directorio/paquete 'control'.
CMD ["python", "-m", "control"]
