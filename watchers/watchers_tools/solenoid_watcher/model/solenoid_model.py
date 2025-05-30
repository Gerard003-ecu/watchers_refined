#!/usr/bin/env python3
"""
solenoid_model.py

Este módulo implementa un modelo físico para simular la evolución del campo magnético
dentro de un solenoide helicoidal. Se utiliza un sistema de coordenadas cilíndricas y se
resuelve un conjunto simplificado de ecuaciones diferenciales para modelar la dinámica del
campo axial (B_z) y radial (B_r).
"""

import numpy as np
from scipy.integrate import odeint

def solenoid_model(y, t, I, n, R):
    """
    Modelo simplificado del comportamiento del campo magnético dentro de un solenoide.
    
    Parámetros:
      y : vector de estado [B_r, B_z]
      t : tiempo (requerido por odeint, aunque no se usa explícitamente)
      I : corriente (amperios)
      n : densidad de vueltas (vueltas por metro)
      R : radio del solenoide (metros)
    
    Retorna:
      dydt : derivadas [dB_r/dt, dB_z/dt]
    """
    mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío
    B_r, B_z = y
    dB_r_dt = -B_r / 0.1  # Relajación rápida de la componente radial
    dB_z_dt = (mu0 * n * I - B_z) / 0.1  # Relajación hacia el valor teórico
    return [dB_r_dt, dB_z_dt]

def simulate_solenoid(I, n, R, initial_state=[0, 0], t_end=1.0, num_points=100):
    if I < 0 or n < 0 or R <= 0:
        raise ValueError("Parámetros I, n, R deben ser positivos")
    """
    Simula la evolución del campo magnético en el solenoide.
    
    Parámetros:
      I : corriente (amperios)
      n : densidad de vueltas (vueltas por metro)
      R : radio del solenoide (metros)
      initial_state : estado inicial [B_r, B_z]
      t_end : tiempo final de la simulación (segundos)
      num_points : número de puntos de tiempo a calcular
    
    Retorna:
      t : vector de tiempos
      sol : solución de las ecuaciones en cada instante
    """
    t = np.linspace(0, t_end, num_points)
    sol = odeint(solenoid_model, initial_state, t, args=(I, n, R))
    return t, sol
