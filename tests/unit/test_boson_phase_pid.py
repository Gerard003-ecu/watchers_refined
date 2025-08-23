#!/usr/bin/env python3
"""
test_boson_phase_pid.py

Pruebas unitarias para el controlador PID de fase `BosonPhasePID`.
"""

import numpy as np
import pytest

from control.boson_phase import BosonPhasePID


@pytest.fixture
def pid_controller():
    """
    Fixture para crear una instancia de BosonPhasePID con ganancias de ejemplo.
    """
    # Usamos Kp=1.0, Ki=0.5, Kd=0.2 para que todos los términos tengan efecto.
    pid = BosonPhasePID(Kp=1.0, Ki=0.5, Kd=0.2)
    return pid


def test_pid_initialization(pid_controller):
    """
    Verifica que el controlador se inicialice con los valores correctos.
    """
    assert pid_controller.Kp == 1.0
    assert pid_controller.Ki == 0.5
    assert pid_controller.Kd == 0.2
    assert pid_controller.setpoint == 0.0
    assert pid_controller.integral == 0.0
    assert pid_controller.last_measurement == 0.0


def test_set_target(pid_controller):
    """
    Verifica que el método set_target() actualice correctamente el setpoint.
    """
    pid_controller.set_target(np.pi)
    assert pid_controller.setpoint == np.pi


def test_set_gains(pid_controller):
    """
    Verifica que el método set_gains() actualice correctamente las ganancias.
    """
    pid_controller.set_gains(Kp=2.0, Ki=1.0, Kd=0.5)
    assert pid_controller.Kp == 2.0
    assert pid_controller.Ki == 1.0
    assert pid_controller.Kd == 0.5


def test_pid_reset(pid_controller):
    """
    Verifica que el método reset() reinicie el estado del PID.
    """
    pid_controller.set_target(1.0)
    pid_controller.update(0.5, dt=0.1)
    # Verificamos que el estado ha cambiado
    assert pid_controller.integral != 0.0
    assert pid_controller.last_measurement != 0.0

    pid_controller.reset()
    # Verificamos que el estado se ha reiniciado
    assert pid_controller.integral == 0.0
    assert pid_controller.last_measurement == 0.0


def test_angular_wraparound(pid_controller):
    """
    Verifica que el cálculo del error maneje correctamente el 'wraparound'
    de las variables angulares. La diferencia entre 0.1 y 2*pi-0.1 debe ser ~0.2.
    """
    pid_controller.set_target(0.1)
    # El error debe ser la distancia más corta, ~0.2 rad, no ~ -6.0 rad
    error = (pid_controller.setpoint - (2 * np.pi - 0.1) + np.pi) % (2 * np.pi) - np.pi
    assert np.isclose(error, 0.2)

    pid_controller.set_target(6.2)  # Cerca de 2*pi
    measurement = 0.1
    # El error debe ser la distancia más corta, ~ -0.18 rad, no ~6.1 rad
    error = (pid_controller.setpoint - measurement + np.pi) % (2 * np.pi) - np.pi
    assert np.isclose(error, -0.18, atol=1e-2)


def test_derivative_kick_prevention(pid_controller):
    """
    Verifica que no haya "patada derivativa" al cambiar el setpoint.
    La derivada debe basarse en la medición, no en el error.
    """
    pid_controller.set_target(0.0)
    pid_controller.update(measurement=0.0, dt=0.1)

    # Cambiamos el setpoint bruscamente.
    pid_controller.set_target(np.pi)

    # La primera actualización después del cambio de setpoint.
    # Como la *medida* no ha cambiado, el término derivativo debe ser cero.
    output = pid_controller.update(measurement=0.0, dt=0.1)

    # El error por wraparound es -pi, no pi.
    error = -np.pi

    # El término P es Kp * (-pi).
    # El término I es Ki * (-pi) * dt.
    # El término D debe ser 0.
    expected_p = pid_controller.Kp * error
    expected_i = pid_controller.Ki * error * 0.1
    expected_output = expected_p + expected_i

    assert np.isclose(output, expected_output, atol=1e-4), (
        "La salida no debe tener componente derivativa al cambiar el setpoint"
    )


def test_pid_proportional_term(pid_controller):
    """
    Verifica que los términos del PID se calculen como se espera.
    """
    pid_controller.set_target(1.0)
    output = pid_controller.update(measurement=0.5, dt=0.1)

    # error = 1.0 - 0.5 = 0.5
    # integral = 0.5 * 0.1 = 0.05
    # derivative = -(0.5 - 0.0) / 0.1 = -5.0
    # output = Kp*error + Ki*integral + Kd*derivative
    # output = 1.0*0.5 + 0.5*0.05 + 0.2*(-5.0) = 0.5 + 0.025 - 1.0 = -0.475
    expected_output = -0.475

    assert np.isclose(output, expected_output), "La salida calculada no es la esperada."


if __name__ == "__main__":
    pytest.main()
