#!/usr/bin/env python3
"""
solenoid_controller.py

Este módulo implementa un controlador PID para ajustar la señal de control
del solenoide, utilizando el modelo físico implementado en solenoid_model.
"""

from agent_ai.model.solenoid_model import simulate_solenoid


class SolenoidController:
    def __init__(self, desired_Bz, Kp=1.0, Ki=0.1, Kd=0.05):
        """
        Inicializa el controlador PID para el solenoide.
        
        Parámetros:
          desired_Bz : valor deseado del campo magnético axial (Tesla)
          Kp, Ki, Kd : coeficientes del controlador PID
        """
        self.desired_Bz = desired_Bz
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral_error = 0.0
        self.last_error = 0.0

    def pid_control(self, measured_Bz, dt):
        """
        Calcula la señal de control usando PID.
        
        Parámetros:
          measured_Bz : valor medido del campo axial
          dt : intervalo de tiempo
          
        Retorna:
          control_signal : señal de control calculada
        """
        error = self.desired_Bz - measured_Bz
        self.integral_error += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        self.last_error = error
        control_signal = (self.Kp * error + 
                          self.Ki * self.integral_error + 
                          self.Kd * derivative)
        return control_signal

    def update(self, current, n, R, dt=0.1):
        """
        Ejecuta una simulación del modelo del solenoide y actualiza la señal.
        
        Parámetros:
          current : corriente actual (amperios)
          n : densidad de vueltas (vueltas por metro)
          R : radio del solenoide (metros)
          dt : intervalo de tiempo para la simulación
        
        Retorna:
          control_signal : señal de control calculada
          measured_Bz : valor medido del campo axial (Tesla)
        """
        t, sol = simulate_solenoid(
            current, n, R, initial_state=[0, 0], t_end=dt, num_points=2
        )
        # Tomamos el último valor de B_z
        measured_Bz = sol[-1, 1]
        control_signal = self.pid_control(measured_Bz, dt)
        return control_signal, measured_Bz


if __name__ == "__main__":
    desired_Bz = 1e-3  # Tesla
    controller = SolenoidController(desired_Bz, Kp=1000, Ki=50, Kd=10)
    current = 5.0    # Corriente en amperios
    n = 1000   # Vueltas por metro
    R = 0.05   # Radio en metros
    dt = 0.1
    control_signal, measured_Bz = controller.update(current, n, R, dt)
    message = (f"Señal de control: {control_signal:.3f}, "
               f"Bz medido: {measured_Bz:.6f} T")
    print(message)