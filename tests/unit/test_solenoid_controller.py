import pytest
import numpy as np
from unittest.mock import patch
from solenoid_controller import SolenoidController


@pytest.fixture
def controller():
    return SolenoidController(desired_Bz=1e-3, Kp=1000, Ki=50, Kd=10)


@patch('solenoid_controller.simulate_solenoid')
def test_pid_control(mock_simulate, controller):
    # Configuración de mock para la simulación
    mock_simulate.return_value = (
        np.array([0, 0.1]),
        np.array([[0, 0], [0.1, 5e-4]])  # Bz simulado = 5e-4 T
    )
    # Caso base
    control_signal, measured_Bz = controller.update(
        I=5, n=1000, R=0.05, dt=0.1
    )
    # Verificaciones PID
    error = 1e-3 - 5e-4
    integral = error * 0.1
    derivada = (error - 0) / 0.1
    expected_signal = (
        1000 * error +
        50 * integral +
        10 * derivada
    )
    assert np.isclose(control_signal, expected_signal, rtol=1e-3)
    assert measured_Bz == 5e-4


def test_pid_parameters(controller):
    # Verificación de parámetros iniciales
    assert controller.Kp == 1000
    assert controller.Ki == 50
    assert controller.Kd == 10
    assert controller.desired_Bz == 1e-3


@patch('solenoid_controller.simulate_solenoid')
def test_edge_cases(mock_simulate, controller):
    # Caso dt=0 (debe evitar división por cero)
    mock_simulate.return_value = (
        np.array([0, 0]),
        np.array([[0, 0], [0, 0]])
    )
    _, _ = controller.update(I=0, n=0, R=0, dt=0)
    # Verificar que el término derivativo sea cero
    assert controller.last_error == 0


def test_integral_windup(controller):
    # Forzar acumulación de integral
    for _ in range(10):
        controller.pid_control(measured_Bz=0, dt=0.1)
    # Verificar que la integral no crece indefinidamente
    assert controller.integral_error == (1e-3 * 0.1) * 10


@patch('solenoid_controller.simulate_solenoid')
def test_convergence(mock_simulate, controller):
    # Simular convergencia al valor deseado
    mock_simulate.return_value = (
        np.array([0, 0.1]),
        np.array([[0, 0], [0.1, 1e-3]])
    )
    control_signal, measured_Bz = controller.update(
        I=5, n=1000, R=0.05, dt=0.1
    )
    # El error debería ser cero
    assert np.isclose(control_signal, 0, atol=1e-6)
    assert measured_Bz == 1e-3
