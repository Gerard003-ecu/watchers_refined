import numpy as np
import pytest
from watchers.watchers_tools.solenoid_watcher.controller.solenoid_controller import (
    capturar_imagen,
    procesar_imagen,
    generar_retroalimentacion,
    retroalimentacion_optica
)


def test_capturar_imagen():
    """
    Prueba la función `capturar_imagen`.
    Verifica que la función convierte
    correctamente una matriz de estado
    en una imagen en escala de grises (0-255).
    """
    # Simulamos una matriz de estado con valores entre 0 y 1.
    state = np.array([[0.2, 0.5], [0.7, 1.0]])
    image = capturar_imagen(state)
    # La imagen debe tener la misma forma que la matriz y valores en 0-255.
    assert image.shape == state.shape, (
        "La imagen debe conservar la forma de la matriz de estado."
    )
    assert image.dtype == np.uint8, "La imagen debe estar en formato uint8."
    assert (image.min() >= 0 and image.max() <= 255), (
        "Los valores de la imagen deben estar en [0,255]."
    )


def test_procesar_imagen():
    """
    Prueba la función `procesar_imagen`.
    Verifica que la función refleja
    horizontalmente la imagen correctamente.
    """
    # Creamos una imagen de ejemplo.
    image = np.array([[0, 50], [100, 150]], dtype=np.uint8)
    processed = procesar_imagen(image)
    # Al reflejar horizontalmente, la imagen debe quedar invertida.
    expected = np.fliplr(image)
    np.testing.assert_array_equal(processed, expected)


def test_generar_retroalimentacion():
    """
    Prueba la función `generar_retroalimentacion`.
    Verifica que genera una señal
    de retroalimentación normalizada (0-1).
    """
    # Creamos una imagen con valores fijos para predecir el feedback.
    image = np.full((10, 10), 128, dtype=np.uint8)
    feedback = generar_retroalimentacion(image)
    # Feedback esperado: valor medio normalizado.
    expected = 128 / 255.0
    assert abs(feedback - expected) < 1e-3, (
        f"Feedback esperado {expected}, obtenido {feedback}"
    )


def test_retroalimentacion_optica():
    """
    Prueba la función `retroalimentacion_optica`.
    Verifica que integra correctamente
    captura, procesamiento y generación.
    """
    # Simulamos una matriz de estado.
    state = np.array([[0.1, 0.9], [0.4, 0.7]])
    feedback = retroalimentacion_optica(state)
    # El feedback generado debe estar entre 0 y 1.
    assert 0 <= feedback <= 1, "Feedback debe estar normalizado entre 0 y 1."


if __name__ == "__main__":
    pytest.main()
