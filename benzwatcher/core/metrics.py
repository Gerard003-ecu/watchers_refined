#!/usr/bin/env python3
"""
metrics.py

Define las métricas para monitorear el desempeño de BenzWatcher.
Estas métricas pueden ser recogidas por Prometheus y visualizadas en Grafana.
"""

from prometheus_client import Counter, Histogram

# Contador para las reacciones catalíticas exitosas
REACTIVATION_SUCCESS = Counter(
    'benzwatcher_success_total',
    'Total de reacciones catalíticas exitosas'
)

# Contador para las reacciones fallidas
REACTIVATION_FAILURE = Counter(
    'benzwatcher_failure_total',
    'Total de reacciones catalíticas fallidas'
)

# Histograma para medir el tiempo de reacción (en segundos)
REACTION_TIME = Histogram(
    'benzwatcher_reaction_time_seconds',
    'Tiempo de reacción en BenzWatcher'
)

def record_success():
    REACTIVATION_SUCCESS.inc()

def record_failure():
    REACTIVATION_FAILURE.inc()

def observe_reaction_time(duration: float):
    REACTION_TIME.observe(duration)

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    # Simular una reacción
    time.sleep(0.2)
    duration = time.perf_counter() - start
    observe_reaction_time(duration)
    record_success()
    print("Métricas actualizadas.")
