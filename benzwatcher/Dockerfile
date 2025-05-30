FROM python:3.9-slim

# Establecer directorio de trabajo
WORKDIR /app/benzwatcher

# Copiar requirements.txt y luego instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos
COPY . .

# Exponer el puerto en el que FastAPI correrá (en este ejemplo 8000)
EXPOSE 8000

# Comando de inicio: ejecutar la API de BenzWatcher
CMD ["uvicorn", "api.endpoints:app", "--host", "0.0.0.0", "--port", "8000"]
