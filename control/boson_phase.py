#!/usr/bin/env python3
"""
boson_phase.py

Este módulo implementa un controlador PID de calidad de producción,
optimizado para variables angulares como la fase (en radianes).
La clase BosonPhasePID gestiona el control de la fase para la matriz ECU,
asegurando estabilidad y robustez.
"""

import logging
import numpy as np

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"
)


class BosonPhasePID:
    """
    Controlador PID para variables angulares (fase en radianes).

    Implementa un controlador Proporcional-Integral-Derivativo (PID) con
    mejoras específicas para manejar la naturaleza cíclica de los ángulos,
    prevenir la "patada derivativa" y ofrecer una interfaz clara para su
    configuración en tiempo de real.
    """

    def __init__(self, Kp, Ki, Kd, integral_max=100.0, integral_min=-100.0):
        """
        Inicializa el controlador PID.

        Args:
            Kp (float): Ganancia proporcional.
            Ki (float): Ganancia integral.
            Kd (float): Ganancia derivativa.
            integral_max (float, opcional): Límite superior para el término
                                            integral (anti-windup).
            integral_min (float, opcional): Límite inferior para el término
                                            integral (anti-windup).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = 0.0  # El setpoint se establece con set_target
        self.integral = 0.0
        self.last_error = 0.0
        self.last_measurement = 0.0
        self.integral_max = integral_max
        self.integral_min = integral_min
        logging.debug(
            f"PID inicializado: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}"
        )

    def set_target(self, new_setpoint):
        """
        Establece el valor objetivo (setpoint) que el controlador debe alcanzar.

        Args:
            new_setpoint (float): El nuevo valor deseado para la variable
                                  controlada, en radianes.
        """
        self.setpoint = new_setpoint
        logging.debug(f"Nuevo setpoint establecido: {self.setpoint}")

    def set_gains(self, Kp, Ki, Kd):
        """
        Ajusta las ganancias del controlador PID en tiempo de ejecución.

        Esto es útil para el ajuste automático (auto-tuning) o para adaptar
        el comportamiento del controlador a diferentes condiciones operativas.

        Args:
            Kp (float): Nueva ganancia proporcional.
            Ki (float): Nueva ganancia integral.
            Kd (float): Nueva ganancia derivativa.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        logging.info(
            f"Gancias actualizadas: Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}"
        )

    def update(self, measurement, dt):
        """
        Calcula la salida del controlador PID para la medida actual.

        Args:
            measurement (float): El valor medido actualmente (en radianes).
            dt (float): El intervalo de tiempo transcurrido desde la última
                        actualización, en segundos.

        Returns:
            float: La salida del controlador PID, que representa la acción
                   de control a aplicar.
        """
        if dt <= 0:
            logging.warning(
                "El intervalo de tiempo dt debe ser positivo. "
                "Se usará dt=1.0 por defecto para evitar división por cero."
            )
            dt = 1.0

        # 1. Cálculo de error angular (manejo de "wraparound")
        error = (self.setpoint - measurement + np.pi) % (2 * np.pi) - np.pi

        # 2. Término integral con anti-windup
        self.integral += error * dt
        self.integral = max(
            min(self.integral, self.integral_max), self.integral_min
        )

        # 3. Término derivativo sobre la medición (previene "derivative kick")
        derivative = -(measurement - self.last_measurement) / dt

        # 4. Calcular la salida del controlador
        output = (self.Kp * error +
                  self.Ki * self.integral +
                  self.Kd * derivative)

        # 5. Actualizar estado para el próximo ciclo
        self.last_error = error
        self.last_measurement = measurement

        logging.debug(
            f"update -> err: {error:.3f}, int: {self.integral:.3f}, "
            f"der: {derivative:.3f}, out: {output:.3f}"
        )

        return output

    def reset(self):
        """
        Reinicia el estado interno del controlador PID.

        Pone a cero el término integral, el último error y la última medición,
        lo que es útil al iniciar una nueva secuencia de control o tras un
        cambio brusco en el sistema.
        """
        self.integral = 0.0
        self.last_error = 0.0
        self.last_measurement = 0.0
        logging.info("PID reset: acumuladores y estado reiniciados.")


if __name__ == "__main__":
    # Prueba simple del controlador PID para fase
    controller = BosonPhasePID(Kp=1.0, Ki=0.1, Kd=0.05)
    controller.set_target(np.pi / 2)  # Objetivo: 90 grados

    print("Prueba de BosonPhasePID para control de fase:")
    measurement = 0.0
    dt = 0.1
    for i in range(10):
        output = controller.update(measurement, dt)
        print(
            f"Paso {i}: Medida={measurement:.2f}, "
            f"Objetivo={controller.setpoint:.2f}, Salida={output:.3f}"
        )
        # Simulación simple: la salida afecta a la siguiente medida
        measurement += output * dt
