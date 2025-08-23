"""
Propósito:
Cuantifica la capacidad útil de trabajo del sistema para priorizar acciones de
seguridad mediante principios termodinámicos.

Métricas Operativas:

- Entropía de Seguridad: Desorden en señales de monitoreo [Umbral: > 2.5 bits]
- Pérdida de Exergía: Eficiencia termodinámica degradada [Umbral: > 15%]
- Tiempo de Respuesta: Latencia en ejecutar protocolos [Umbral: < 200 ms]
"""

import numpy as np


class ExergyModel:
    def __init__(self, environment_temp=300):  # 300K = 27°C
        self.T0 = environment_temp  # Temperatura ambiente de referencia
        self.R = 8.314  # Constante de los gases [J/mol·K]

    def calculate_exergy(self, system_state):
        """Calcula la exergía como máximo trabajo útil obtenible"""
        # U: Energía interna, S: Entropía, p: presión, V: volumen
        U = system_state["energy"]
        S = system_state["entropy"]
        p = system_state["pressure"]
        V = system_state["volume"]
        exergy = U + p * V - self.T0 * S
        return exergy

    def exergy_priority(self, anomaly_vector):
        """Prioriza acciones basado en pérdida exergética potencial"""
        entropy_component = np.abs(anomaly_vector[0])
        energy_component = np.linalg.norm(anomaly_vector[1:3])
        return entropy_component * energy_component

    def optimize_energy_distribution(self, vector):
        """Maximiza exergía mediante optimización convexa"""
        from scipy.optimize import minimize

        def exergy_loss(x):
            return -np.dot(x, vector) + 0.5 * np.linalg.norm(x) ** 2

        constraints = {"type": "ineq", "fun": lambda x: np.sum(x) - 0.1}
        result = minimize(
            exergy_loss,
            x0=np.ones_like(vector),
            method="SLSQP",
            constraints=constraints,
        )
        return result.x

    def calculate_security_entropy(self, signals):
        """Entropía de Shannon aplicada a métricas de seguridad"""
        probabilities = np.abs(signals) / np.sum(np.abs(signals))
        return -np.sum(probabilities * np.log2(probabilities + 1e-12))
