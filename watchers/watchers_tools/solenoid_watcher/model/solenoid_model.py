#!/usr/bin/env python3
"""
solenoid_model.py

Este módulo implementa un modelo físico para simular la evolución del campo
magnético dentro de un solenoide helicoidal. Se utiliza un sistema de
coordenadas cilíndricas y se resuelve un conjunto simplificado de ecuaciones
diferenciales para modelar la dinámica del campo axial (B_z) y radial (B_r).
"""

import numpy as np
from scipy.integrate import odeint


def solenoid_model(state, time, current, turns_density, radius):
    """
    Modelo simplificado del comportamiento del campo magnético dentro de un
    solenoide.

    Parámetros:
      state : vector de estado [B_r, B_z]
      time : tiempo (requerido por odeint)
      current : corriente (amperios)
      turns_density : densidad de vueltas (vueltas por metro)
      radius : radio del solenoide (metros)

    Retorna:
      dydt : derivadas [dB_r/dt, dB_z/dt]
    """
    mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío
    b_radial, b_axial = state
    
    # Relajación rápida de la componente radial
    db_radial_dt = -b_radial / 0.1
    
    # Relajación hacia el valor teórico
    db_axial_dt = (mu0 * turns_density * current - b_axial) / 0.1
    
    return [db_radial_dt, db_axial_dt]


def simulate_solenoid(
        current, turns_density, radius,
        initial_state=(0, 0), t_end=1.0, num_points=100):
    """
    Simula la evolución del campo magnético en el solenoide.

    Parámetros:
      current : corriente (amperios)
      turns_density : densidad de vueltas (vueltas por metro)
      radius : radio del solenoide (metros)
      initial_state : estado inicial [B_r, B_z]
      t_end : tiempo final de simulación (segundos)
      num_points : número de puntos de tiempo a calcular

    Retorna:
      t : vector de tiempos
      sol : solución de las ecuaciones en cada instante
    """
    if current < 0 or turns_density < 0 or radius <= 0:
        raise ValueError("Parámetros deben ser positivos")
    
    time = np.linspace(0, t_end, num_points)
    solution = odeint(
        solenoid_model,
        initial_state,
        time,
        args=(current, turns_density, radius)
      )
    return time, solution