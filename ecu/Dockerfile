# 1. Imagen Base: Usa una imagen base de Python específica y slim.
FROM python:3.12-slim-bookworm

# 2. Entorno: Establece WORKDIR y variables de entorno.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Instalar dependencias del sistema (curl para HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Optimización de Caché de Dependencias:
# Copia solo el archivo de requerimientos para aprovechar la caché de Docker.
COPY requirements.txt .

# Instala las dependencias de Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación.
# El .dockerignore debería usarse para excluir archivos innecesarios.
COPY . .

# 4. Seguridad: Crea un usuario no-root y cambia a él.
RUN useradd -m app
USER app

# 5. Metadatos y Ejecución:
# Expone el puerto que el servicio utiliza.
EXPOSE 8000

# Añade una instrucción HEALTHCHECK.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/api/health || exit 1

# Asegúrate de que la instrucción CMD final use el formato "exec".
CMD ["python", "matriz_ecu.py"]
