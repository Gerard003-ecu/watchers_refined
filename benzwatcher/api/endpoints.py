#!/usr/bin/env python3
"""
endpoints.py

Endpoints REST para interactuar con BenzWatcher.
Utiliza FastAPI para exponer:
  - Un endpoint para catalizar una señal.
  - Un endpoint para exponer las métricas.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from benzwatcher.core import metrics
from benzwatcher.core.benz_watcher import BenzWatcher

app = FastAPI(title="BenzWatcher API", version="1.0")

# Instancia global de BenzWatcher
benz_watcher = BenzWatcher(base_value=100.0)


class CatalysisRequest(BaseModel):
    signal: float


class CatalysisResponse(BaseModel):
    adjusted_value: float


@app.post("/api/catalyze", response_model=CatalysisResponse)
def catalyze_signal(request: CatalysisRequest):
    try:
        adjusted = benz_watcher.catalyze(request.signal)
        metrics.record_success()
        return CatalysisResponse(adjusted_value=adjusted)
    except Exception as e:
        metrics.record_failure()
        raise HTTPException(status_code=500, detail=str(e)) from e


# Endpoint para exponer las métricas en formato compatible con Prometheus
@app.get("/metrics")
def get_metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
