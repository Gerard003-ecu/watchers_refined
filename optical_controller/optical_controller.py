#!/usr/bin/env python3
"""
optical_controller.py

Módulo que simula el procesamiento óptico para el sistema Watchers. Captura una "imagen" del estado global,
la procesa (reflejo) y genera una señal de retroalimentación. Proporciona un servicio REST para integrarse
con otros componentes del sistema.

Funcionalidades:
- Captura de imagen a partir de una matriz de estado.
- Procesamiento de imagen (reflejo horizontal).
- Generación de señal de retroalimentación normalizada.
- Endpoint REST para obtener retroalimentación óptica.
- Endpoint de salud para monitoreo.
"""

import numpy as np
import cv2
import logging
from flask import Flask, request, jsonify
from typing import List, Dict, Any

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
    """
    Captura una imagen simulada a partir de una matriz de estado.
    
    Args:
        matriz_estado (List[List[float]]): Matriz de estado del sistema.
    
    Returns:
        np.ndarray: Imagen en escala de grises (0-255).
    """
    try:
        imagen = np.array(matriz_estado, dtype=np.float32)
        imagen = (imagen - np.min(imagen)) / (np.max(imagen) - np.min(imagen) + 1e-6) * 255
        imagen = imagen.astype(np.uint8)
        logger.debug("Imagen capturada del estado.")
        return imagen
    except Exception as e:
        logger.error(f"Error capturando imagen: {str(e)}")
        raise ValueError("Error al procesar la matriz de estado.")

def procesar_imagen(imagen: np.ndarray) -> np.ndarray:
    """
    Aplica un reflejo horizontal a la imagen capturada.
    
    Args:
        imagen (np.ndarray): Imagen en escala de grises.
    
    Returns:
        np.ndarray: Imagen reflejada.
    """
    try:
        imagen_reflejada = cv2.flip(imagen, 1)
        logger.debug("Imagen procesada (reflejada).")
        return imagen_reflejada
    except Exception as e:
        logger.error(f"Error procesando imagen: {str(e)}")
        raise ValueError("Error al aplicar transformación óptica.")

def generar_retroalimentacion(imagen_procesada: np.ndarray) -> float:
    """
    Genera una señal de retroalimentación normalizada a partir de la imagen procesada.
    
    Args:
        imagen_procesada (np.ndarray): Imagen reflejada.
    
    Returns:
        float: Señal de retroalimentación en el rango [0.0, 1.0].
    """
    try:
        retroalimentacion = float(np.mean(imagen_procesada)) / 255.0
        logger.info(f"Retroalimentación generada: {retroalimentacion:.3f}")
        return retroalimentacion
    except Exception as e:
        logger.error(f"Error generando retroalimentación: {str(e)}")
        raise ValueError("Error al calcular la retroalimentación.")

def retroalimentacion_optica(matriz_estado: List[List[float]]) -> float:
    """
    Integra la captura, procesamiento y generación de retroalimentación.
    
    Args:
        matriz_estado (List[List[float]]): Matriz de estado del sistema.
    
    Returns:
        float: Señal de retroalimentación óptica.
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
    """
    Endpoint REST para obtener retroalimentación óptica.
    
    Expects:
        JSON con clave 'estado' (matriz de estado).
    
    Returns:
        jsonify: Respuesta JSON con retroalimentación o mensaje de error.
    """
    try:
        datos = request.get_json()
        if not datos or "estado" not in datos:
            return jsonify({
                "status": "error",
                "message": "Debe enviarse un JSON con la clave 'estado'."
            }), 400
        
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
    """
    Endpoint de salud para monitoreo del módulo.
    
    Returns:
        jsonify: Respuesta JSON con estado del módulo.
    """
    return jsonify({
        "status": "success",
        "module": "Optical_controller",
        "mensaje": "Módulo operativo"
    }), 200

def iniciar_servicio(host: str = "0.0.0.0", port: int = 8001) -> None:
    """
    Inicia el servidor Flask para el controlador óptico.
    
    Args:
        host (str): Dirección IP del host.
        port (int): Puerto de escucha.
    """
    logger.info(f"Iniciando servicio en {host}:{port}")
    app.run(host=host, port=port, debug=False)
    
# Alias para compatibilidad con pruebas y otros módulos que importen nombres en inglés
capture_image = capturar_imagen
process_image = procesar_imagen
generate_feedback = generar_retroalimentacion
optical_feedback = retroalimentacion_optica

if __name__ == "__main__":
    iniciar_servicio()