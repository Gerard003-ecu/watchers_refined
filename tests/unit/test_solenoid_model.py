# tests/test_solenoid_model.py
import numpy as np
import pytest
from agent_ai.model.solenoid_model import simulate_solenoid

def test_simulate_field():
    I = 0.5      # Corriente en amperios
    n = 150      # Densidad de vueltas (vueltas por metro)
    R = 0.2      # Radio en metros (no usado en este modelo básico)
    t_end = 1.0  # Tiempo final de la simulación en segundos
    num_points = 100  # Número de puntos en la simulación
    mu0 = 4 * np.pi * 1e-7  # Permeabilidad del vacío

    # Se simula la evolución del campo magnético
    t, sol = simulate_solenoid(I, n, R, initial_state=[0, 0], t_end=t_end, num_points=num_points)
    
    # El componente Bz es el segundo elemento de cada estado, tomamos el final
    final_Bz = sol[-1, 1]
    expected_Bz = mu0 * n * I
    
    # Se permite una pequeña tolerancia numérica
    assert np.isclose(final_Bz, expected_Bz, atol=1e-8)

if __name__ == "__main__":
    import pytest
    pytest.main()
