FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --upgrade pip setuptools wheel
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido del módulo dashboard
COPY . /app

ENV PYTHONPATH="/app"
EXPOSE 8050

CMD ["python", "dashboard/dashboard.py"]
