#!/usr/bin/env python3
"""Simulador de procesamiento óptico para el sistema Watchers.

Este módulo captura una representación visual del estado global del sistema,
aplica un procesamiento de imagen (reflejo horizontal) y genera una señal
de retroalimentación. Se expone como un servicio REST para facilitar la
integración con otros componentes del sistema Watchers.

Funcionalidades principales:
  - Captura de imagen: Convierte una matriz de estado numérico en una imagen
    en escala de grises.
  - Procesamiento de imagen: Aplica un reflejo horizontal a la imagen capturada.
  - Generación de retroalimentación: Calcula una señal normalizada a partir
    de la imagen procesada.
  - Servicio REST: Ofrece endpoints para obtener la retroalimentación óptica
    y verificar el estado de salud del módulo.
"""

import numpy as np
import cv2
import logging
from flask import Flask, request, jsonify
from typing import List

# Configuración centralizada
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/optical_controller.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("optical_controller")

app = Flask(__name__)


def capturar_imagen(matriz_estado: List[List[float]]) -> np.ndarray:
    """Convierte una matriz de estado en una imagen en escala de grises.

    La matriz de estado se normaliza para que sus valores se encuentren
    en el rango de 0 a 255 y luego se convierte a un tipo de dato de 8 bits
    sin signo, adecuado para representaciones de imagen.

    Args:
        matriz_estado: Una lista de listas de flotantes representando
            el estado del sistema. Se espera que sea una matriz 2D.

    Returns:
        Un array de NumPy (np.ndarray) que representa la imagen generada
        en escala de grises (valores de 0 a 255).

    Raises:
        ValueError: Si ocurre un error durante el procesamiento de la
            matriz de estado (por ejemplo, si no es una matriz válida).
    """
    try:
        imagen = np.array(matriz_estado, dtype=np.float32)
        imagen = (
            imagen - np.min(imagen)
        ) / (np.max(imagen) - np.min(imagen) + 1e-6) * 255
        imagen = imagen.astype(np.uint8)
        logger.debug("Imagen capturada del estado.")
        return imagen
    except Exception as e:
        logger.error(f"Error capturando imagen: {str(e)}")
        raise ValueError("Error al procesar la matriz de estado.")


def procesar_imagen(imagen: np.ndarray) -> np.ndarray:
    """Aplica un procesamiento de reflejo horizontal a una imagen.

    Utiliza la función `flip` de OpenCV para invertir la imagen a lo largo
    del eje vertical (reflejo horizontal).

    Args:
        imagen: Un array de NumPy (np.ndarray) que representa la imagen
            de entrada en escala de grises.

    Returns:
        Un array de NumPy (np.ndarray) que representa la imagen con el
        reflejo horizontal aplicado.

    Raises:
        ValueError: Si ocurre un error durante la transformación de la imagen.
    """
    try:
        imagen_reflejada = cv2.flip(imagen, 1)
        logger.debug("Imagen procesada (reflejada).")
        return imagen_reflejada
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        raise ValueError("Error al aplicar transformación óptica.")


def generar_retroalimentacion(imagen_procesada: np.ndarray) -> float:
    """Calcula una señal de retroalimentación normalizada desde una imagen.

    La señal se obtiene calculando el valor promedio de los píxeles de la
    imagen y normalizándolo al rango [0.0, 1.0] dividiendo por 255 (el valor
    máximo posible para un píxel en una imagen de 8 bits en escala de grises).

    Args:
        imagen_procesada: Un array de NumPy (np.ndarray) que representa la
            imagen procesada (por ejemplo, reflejada) en escala de grises.

    Returns:
        Un valor flotante que representa la señal de retroalimentación
        normalizada, en el rango de 0.0 a 1.0.

    Raises:
        ValueError: Si ocurre un error al calcular la retroalimentación.
    """
    try:
        retroalimentacion = float(np.mean(imagen_procesada)) / 255.0
        logger.info(f"Retroalimentación generada: {retroalimentacion:.3f}")
        return retroalimentacion
    except Exception as e:
        logger.error(f"Error generando retroalimentación: {str(e)}")
        raise ValueError("Error al calcular la retroalimentación.")


def retroalimentacion_optica(matriz_estado: List[List[float]]) -> float:
    """Procesa una matriz de estado para generar una señal de retroalimentación óptica.

    Este es el flujo principal que integra la captura de la imagen desde la
    matriz de estado, el procesamiento de dicha imagen (reflejo), y la
    generación final de una señal de retroalimentación normalizada.

    Args:
        matriz_estado: Una lista de listas de flotantes que representa el
            estado actual del sistema.

    Returns:
        Un valor flotante que es la señal de retroalimentación óptica
        resultante, normalizada entre 0.0 y 1.0.

    Raises:
        ValueError: Si ocurre algún error en cualquiera de los pasos
            (captura, procesamiento o generación de retroalimentación).
    """
    try:
        imagen = capturar_imagen(matriz_estado)
        imagen_procesada = procesar_imagen(imagen)
        return generar_retroalimentacion(imagen_procesada)
    except Exception as e:
        logger.error(f"Error en retroalimentación óptica: {str(e)}")
        raise


@app.route("/api/optical-feedback", methods=["POST"])
def obtener_retroalimentacion() -> jsonify:
    """Endpoint REST para obtener la señal de retroalimentación óptica.

    Este endpoint recibe una matriz de estado del sistema en formato JSON,
    la procesa utilizando el flujo de `retroalimentacion_optica`, y devuelve
    la señal de retroalimentación resultante.

    JSON Request Body:
        estado (List[List[float]]): Matriz bidimensional de números flotantes
            que representa el estado actual del sistema.

    Returns:
        Una respuesta JSON (jsonify) que contiene:
        - En caso de éxito:
            {
                "status": "success",
                "optical_feedback": <float>
            }
        - En caso de error de validación (ej. 'estado' no presente o mal formado):
            {
                "status": "error",
                "message": "<descripción del error>"
            }
            (HTTP 400)
        - En caso de error interno del servidor:
            {
                "status": "error",
                "message": "Error interno del servidor."
            }
            (HTTP 500)
    """
    try:
        datos = request.get_json()
        if not datos or "estado" not in datos:
            return jsonify(
                {
                    "status": "error",
                    "message": "Debe enviarse un JSON con la clave 'estado'.",
                }
            ), 400

        matriz_estado = datos["estado"]
        retroalimentacion = retroalimentacion_optica(matriz_estado)
        return jsonify({
            "status": "success",
            "optical_feedback": retroalimentacion
        })
    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error en endpoint /api/optical-feedback: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error interno del servidor."
        }), 500


@app.route("/api/health", methods=["GET"])
def salud() -> jsonify:
    """Endpoint de chequeo de salud para el módulo Optical Controller.

    Proporciona una respuesta simple para indicar que el servicio está
    funcionando correctamente. Útil para sistemas de monitoreo.

    Returns:
        Una respuesta JSON (jsonify) con el estado del módulo:
        {
            "status": "success",
            "module": "Optical_controller",
            "mensaje": "Módulo operativo"
        }
        (HTTP 200)
    """
    return jsonify({
        "status": "success",
        "module": "Optical_controller",
        "mensaje": "Módulo operativo"
    }), 200


def iniciar_servicio(host: str = "0.0.0.0", port: int = 8001) -> None:
    """Inicia el servidor Flask para el servicio del controlador óptico.

    Este servidor expone los endpoints REST definidos en la aplicación Flask,
    permitiendo la interacción con el controlador óptico.

    Args:
        host: La dirección IP en la que el servidor escuchará.
            Por defecto es "0.0.0.0", lo que significa que escuchará en todas
            las interfaces de red disponibles.
        port: El número de puerto en el que el servidor escuchará.
            Por defecto es 8001.
    """
    logger.info(f"Iniciando servicio en {host}:{port}")
    app.run(host=host, port=port, debug=False)


# Alias para compatibilidad con pruebas y otros módulos que importen nombres
# en inglés
capture_image = capturar_imagen
process_image = procesar_imagen
generate_feedback = generar_retroalimentacion
optical_feedback = retroalimentacion_optica

if __name__ == "__main__":
    iniciar_servicio()
