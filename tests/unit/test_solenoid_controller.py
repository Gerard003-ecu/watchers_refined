from unittest.mock import patch

import numpy as np
import pytest

from watchers.watchers_tools.solenoid_watcher.controller.solenoid_controller import (
    SolenoidController,
)


@pytest.fixture
def controller():
    return SolenoidController(desired_Bz=1e-3, Kp=1000, Ki=50, Kd=10)


@patch(
    "watchers.watchers_tools.solenoid_watcher.controller."
    "solenoid_controller.simulate_solenoid"
)
def test_pid_control(mock_simulate, controller):
    # Configuración de mock para la simulación
    mock_simulate.return_value = (
        np.array([0, 0.1]),
        np.array([[0, 0], [0.1, 5e-4]]),  # Bz simulado = 5e-4 T
    )
    # Caso base
    control_signal, measured_Bz = controller.update(i_amps=5, n=1000, R=0.05, dt=0.1)
    # Verificaciones PID
    error = 1e-3 - 5e-4
    integral = error * 0.1
    derivada = (error - 0) / 0.1
    expected_signal = 1000 * error + 50 * integral + 10 * derivada
    assert np.isclose(control_signal, expected_signal, rtol=1e-3)
    assert measured_Bz == 5e-4


def test_pid_parameters(controller):
    # Verificación de parámetros iniciales
    assert controller.Kp == 1000
    assert controller.Ki == 50
    assert controller.Kd == 10
    assert controller.desired_Bz == 1e-3


@patch(
    "watchers.watchers_tools.solenoid_watcher.controller."
    "solenoid_controller.simulate_solenoid"
)
def test_edge_cases(mock_simulate, controller):
    # Caso dt=0 (debe evitar división por cero)
    mock_simulate.return_value = (np.array([0, 0]), np.array([[0, 0], [0, 0]]))
    _, _ = controller.update(i_amps=0, n=0, R=0, dt=0)
    # Verificar que el término derivativo sea cero
    # last_error will be the current error (desired_Bz - measured_Bz).
    # If measured_Bz is 0 (as from the mock_simulate setup), error is desired_Bz.
    assert controller.last_error == controller.desired_Bz


def test_integral_windup(controller):
    # Forzar acumulación de integral
    for _ in range(10):
        controller.pid_control(measured_Bz=0, dt=0.1)
    # Verificar que la integral no crece indefinidamente
    # Usar pytest.approx para comparaciones de punto flotante
    assert controller.integral_error == pytest.approx((1e-3 * 0.1) * 10)


@patch(
    "watchers.watchers_tools.solenoid_watcher.controller."
    "solenoid_controller.simulate_solenoid"
)
def test_convergence(mock_simulate, controller):
    # Simular convergencia al valor deseado
    mock_simulate.return_value = (np.array([0, 0.1]), np.array([[0, 0], [0.1, 1e-3]]))
    control_signal, measured_Bz = controller.update(i_amps=5, n=1000, R=0.05, dt=0.1)
    # El error debería ser cero
    assert np.isclose(control_signal, 0, atol=1e-6)
    assert measured_Bz == 1e-3
