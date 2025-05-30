# cogniboard/Dockerfile
FROM python:3.9-slim

# Establecer el directorio de trabajo para la aplicación
WORKDIR /app/cogniboard

# Copiar el archivo de requerimientos y aprovechar la caché de Docker:
# Si requirements.txt no cambia, esta capa se reutiliza y no se vuelven a descargar las dependencias.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente del módulo
COPY . .

# Exponer el puerto para el healthcheck (por ejemplo, 9100)
EXPOSE 9100

# Comando de inicio: ejecutar el script monitor.py
CMD ["python", "monitor.py"]
