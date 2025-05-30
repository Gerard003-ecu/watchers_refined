FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Actualiza la lista de paquetes e instala las dependencias del sistema necesarias
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Establecer un timeout mayor para pip (opcional)
ENV PIP_DEFAULT_TIMEOUT=100

# Copiar el archivo unificado de requerimientos para aprovechar la caché de Docker
COPY requirements.txt ./

# Actualizar pip y luego instalar las dependencias sin utilizar la caché interna de pip
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código fuente
COPY . .

# Comando para ejecutar los tests de integración
CMD ["pytest", "tests/integration_tests.py"]
