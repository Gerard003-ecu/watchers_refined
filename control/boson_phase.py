#!/usr/bin/env python3
"""
boson_phase.py

Este módulo implementa un controlador PID adaptativo llamado "BosonPhase"
para armonizar el estado de la matriz ECU (Experiencia Cuántica Unificada), es decir,
para ajustar el estado ideal de la malla vectorial (matriz_ecu).

El controlador PID se ajusta automáticamente según la diferencia entre la medida
actual y el setpoint deseado.
"""

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)


class BosonPhase:
    def __init__(
        self, Kp, Ki, Kd, setpoint=0.0, integral_max=100.0, integral_min=-100.0
    ):
        """
        Inicializa el controlador PID adaptativo.

        Args:
            Kp (float): Ganancia proporcional.
            Ki (float): Ganancia integral.
            Kd (float): Ganancia derivativa.
            setpoint (float, opcional): Valor deseado. Por defecto 0.0.
            integral_max (float, opcional): Límite superior para el término integral.
            integral_min (float, opcional): Límite inferior para el término integral.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.last_error = None
        self.integral_max = integral_max
        self.integral_min = integral_min
        logging.debug(
            f"PID inicializado: Kp={Kp}, Ki={Ki}, Kd={Kd}, setpoint={setpoint}"
        )

    def update(self, measurement, dt):
        """
        Calcula la salida del controlador PID para la medida actual.

        Args:
            measurement (float): El valor medido actualmente.
            dt (float): Intervalo de tiempo transcurrido.

        Returns:
            float: La salida del PID.
        """
        if dt <= 0:
            logging.warning(
                "Intervalo de tiempo dt debe ser mayor a cero. Se usará dt=1.0 por defecto."
            )
            dt = 1.0

        error = self.setpoint - measurement
        self.integral += error * dt
        # Saturar el término integral
        self.integral = max(
            min(self.integral, self.integral_max), self.integral_min
        )
        derivative = 0.0
        if self.last_error is not None:
            derivative = (error - self.last_error) / dt
        self.last_error = error

        output = (
            self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        )
        logging.debug(
            f"update() -> error: {error:.3f}, integral: {self.integral:.3f}, derivative: {derivative:.3f}, output: {output:.3f}"
        )

        # Ajuste adaptativo simple: aumentar Kp si el error es grande
        if abs(error) > 0.1 * self.setpoint:
            self.Kp += 0.01
            logging.info(f"Ajuste adaptativo: Kp incrementado a {self.Kp:.3f}")

        return output

    def compute(self, setpoint, measurement, dt):
        """
        Actualiza el setpoint y calcula la salida del PID.

        Args:
            setpoint (float): El nuevo valor deseado.
            measurement (float): El valor medido actualmente.
            dt (float): Intervalo de tiempo transcurrido.

        Returns:
            float: La salida del PID.
        """
        self.setpoint = setpoint
        logging.debug(f"compute() -> Nuevo setpoint: {setpoint}")
        return self.update(measurement, dt)

    def reset(self):
        """
        Reinicia los acumuladores del PID.
        """
        self.integral = 0.0
        self.last_error = None
        logging.info("PID reset: acumuladores reiniciados.")


if __name__ == "__main__":
    # Prueba simple del controlador PID adaptativo.
    controller = BosonPhase(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=10.0)
    print("Prueba de BosonPhase (PID adaptativo):")
    for i in range(10):
        # Simula una medida constante de 5.0 y dt=1.0 en cada paso
        output = controller.compute(10.0, measurement=5.0, dt=1.0)
        print(f"Paso {i}: salida = {output}")
