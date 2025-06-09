#!/usr/bin/env python3
"""
benz_watcher.py

Implementa la clase BenzWatcher que simula una reacción catalítica inspirada
en la estructura hexagonal del benceno. Esta clase procesa una señal de
entrada y ajusta un valor base en función de una transformación no lineal,
que modela la cinética de la reacción catalítica.
"""

import math


class BenzWatcher:
    def __init__(self, base_value: float):
        """
        Inicializa BenzWatcher con un valor base.

        :param base_value: Valor inicial (por ejemplo, un parámetro operativo).
        """
        self.base_value = base_value

    def catalyze(self, signal: float) -> float:
        """
        Procesa la señal de control para ajustar el valor base.

        Se utiliza una función sigmoidal para modular la señal, lo que
        significa que para señales bajas se realiza poco ajuste, y para
        señales altas se incrementa de forma no lineal.

        :param signal: La señal de control (e.g., generada por un PID).
        :return: Valor ajustado.
        """
        # Función sigmoidal para transformar la señal
        factor = 1 / (1 + math.exp(-signal))
        adjusted_value = self.base_value * factor
        return adjusted_value


if __name__ == "__main__":
    watcher = BenzWatcher(base_value=100.0)
    for s in [-50, -20, 0, 20, 50]:
        print(f"Señal: {s:>4} -> Valor ajustado: {watcher.catalyze(s):.2f}")
