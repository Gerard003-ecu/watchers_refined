#!/usr/bin/env python3
"""
test_boson_phase.py

Pruebas unitarias para el controlador PID adaptativo "BosonPhase".
# Se utiliza una importación directa para obtener la clase desde el
# módulo correspondiente.
"""

import pytest
from control.boson_phase import BosonPhase


@pytest.fixture
def pid_controller():
    # Instanciamos el PID adaptativo "BosonPhase" con parámetros de ejemplo.
    pid = BosonPhase(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=10.0)
    return pid


def test_pid_output_constant_setpoint(pid_controller):
    """
    Verifica que para un setpoint y medida constantes,
    el PID genere una señal de control cercana a 0.
    """
    setpoint = 10.0
    measurement = 10.0
    dt = 0.1
    output = pid_controller.compute(setpoint, measurement, dt)
    error_message = f"Salida inesperada para error nulo: {output}"
    assert abs(output) < 1e-3, error_message


def test_pid_response_to_step(pid_controller):
    """
    Verifica la respuesta del PID ante un cambio abrupto (step) en el setpoint.
    """
    setpoint_final = 5.0
    measurement = 0.0
    dt = 0.1

    pid_controller.reset()
    outputs = []
    for _ in range(10):
        output = pid_controller.compute(setpoint_final, measurement, dt)
        outputs.append(output)
        # Simulamos que la medida se aproxima gradualmente al setpoint.
        measurement += 0.5

    # Se espera que al menos una de las salidas sea significativa.
    assert any(abs(o) > 0.1 for o in outputs),"El PID no reaccionó al cambio."


def test_pid_reset(pid_controller):
    """
    Verifica que el método reset() reinicie
    correctamente los acumuladores del PID.
    """
    # Ejecutamos una actualización para modificar los acumuladores.
    pid_controller.compute(10.0, 5.0, 0.1)
    pid_controller.reset()
    assert pid_controller.integral == 0.0, "Integral not reset."
    assert pid_controller.last_error is None, "Previous error not reset."


if __name__ == "__main__":
    pytest.main()
