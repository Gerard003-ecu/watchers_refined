#!/usr/bin/env python3
"""
test_unitarios.py
Suite de pruebas unitarias para los módulos:
- watchers_wave (malla_watcher)
- watcher_focus
"""

import pytest
import math

from watchers.watchers_tools.malla_watcher.utils.cilindro_grafenal import Cell
from watchers.watchers_tools.malla_watcher.malla_watcher import PhosWave
from watchers.watchers_tools.watcher_focus.watcher_focus import update_indicators


##############################
# Pruebas Unitarias: MÓDULO "watchers_wave"
##############################
def test_phoswave_transmision():
    celda_A = Cell(0, 0, amplitude=1.0, phase=0.0)
    celda_B = Cell(0, 0, amplitude=0.0, phase=0.0)
    resonador = PhosWave(
        coef_transmision=0.6,
        coef_reflexion=0.4,
        tipo_onda=PhosWave.__dict__.get("tipo_onda", None) or "FOTON_A",
        lambda_foton=600
    )
    resonador.transmitir(celda_A, celda_B)
    assert celda_B.amplitude > 0, (
        "La celda_B no incrementó su amplitud"
    )
    assert celda_A.amplitude < 1.0, (
        "La celda_A no redujo su amplitud"
    )


##############################
# Pruebas Unitarias: MÓDULO "watcher_focus"
##############################
def test_update_indicators():
    """Verifica que update_indicators retorne un diccionario con las
    claves esperadas.
    """
    resultado = update_indicators(t=1.0, x=1.0, y=0.0, z=0.5)
    assert isinstance(resultado, dict), (
        "El resultado debe ser un diccionario"
    )
    expected_keys = {"t", "x", "y", "z", "phase", "z_error"}
    assert expected_keys.issubset(resultado.keys()), (
        "Faltan claves en el resultado"
    )
    assert math.isclose(
        resultado["phase"], 0.0, abs_tol=1e-6
    ), "La fase calculada es incorrecta"


if __name__ == "__main__":
    pytest.main()
