#!/usr/bin/env python3
"""Módulo que define la simulación de un campo cimático.

Este módulo modela un campo cimático en una topología toroidal. La dinámica
de las ondas en este campo se rige por principios de propagación, disipación
e interferencia, inspirados en el estudio de la cimática (la visualización
de ondas y vibraciones).

El campo es un campo escalar complejo en una grilla 3D toroidal, donde cada
punto (capa, fila, columna) almacena un número complejo que representa
la amplitud y fase de una onda local. La dinámica simula la evolución de
este campo de ondas.

Proporciona una API REST para:
- Obtener un mapa de densidad de energía del campo.
- Recibir influencias externas (perturbaciones de onda).
- Monitorear la salud del servicio.

Una simulación en segundo plano actualiza continuamente la dinámica del campo.
"""

import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request

# --- Configuración del Logging ---
logger = logging.getLogger("matriz_ecu")

# --- Funciones Auxiliares para Configuración ---


def get_env_int(var_name: str, default: int) -> int:
    """Obtiene una variable de entorno como entero con fallback y logging."""
    try:
        return int(os.environ.get(var_name, str(default)))
    except (ValueError, TypeError):
        logger.warning(
            f"Valor inválido para variable de entorno '{var_name}'. "
            f"Usando valor por defecto: {default}"
        )
        return default


def get_env_float(var_name: str, default: float) -> float:
    """Obtiene una variable de entorno como float con fallback y logging."""
    try:
        return float(os.environ.get(var_name, str(default)))
    except (ValueError, TypeError):
        logger.warning(
            f"Valor inválido para variable de entorno '{var_name}'. "
            f"Usando valor por defecto: {default}"
        )
        return default


# --- Constantes Configurables para el Campo Cimático ---
NUM_CAPAS = get_env_int("ECU_NUM_CAPAS", 3)
NUM_FILAS = get_env_int("ECU_NUM_FILAS", 4)
NUM_COLUMNAS = get_env_int("ECU_NUM_COLUMNAS", 5)
DEFAULT_PROPAGATION_COEFF = get_env_float("ECU_DEFAULT_ALPHA", 0.5)
DEFAULT_DISSIPATION_COEFF = get_env_float("ECU_DEFAULT_DAMPING", 0.05)
SIMULATION_INTERVAL = get_env_float("ECU_SIM_INTERVAL", 1.0)
BETA_COUPLING = get_env_float("ECU_BETA_COUPLING", 0.1)


class ToroidalField:
    """
    Representa un campo de ondas cimáticas en una topología toroidal.

    El campo se modela en una grilla 3D (capas x filas x columnas).
    Cada punto de la grilla almacena un número complejo que representa
    la amplitud y fase de la onda en ese punto.
    La dinámica simula la evolución de este campo bajo efectos de
    propagación de onda, acoplamiento entre capas y disipación de energía,
    análogo a la Ecuación de Onda en un medio con disipación.
    """

    def __init__(
        self,
        num_capas: int,
        num_rows: int,
        num_cols: int,
        propagation_coeffs: Optional[List[float]] = None,
        dissipation_coeffs: Optional[List[float]] = None,
    ):
        """
        Inicializa el campo cimático con dimensiones y parámetros físicos por capa.

        Args:
            num_capas (int): Número de capas (dimensión de profundidad).
            num_rows (int): Número de filas (dimensión vertical).
            num_cols (int): Número de columnas (dimensión azimutal).
            propagation_coeffs (Optional[List[float]]): Coeficientes de
                propagación de la onda por capa. Controla la velocidad
                con que la fase de la onda evoluciona.
            dissipation_coeffs (Optional[List[float]]): Coeficientes de
                disipación de la onda por capa. Controla la pérdida
                de energía (amplitud) de la onda.
        """
        if num_capas <= 0 or num_rows <= 0 or num_cols <= 0:
            raise ValueError(
                "Las dimensiones (capas, filas, columnas) deben ser positivas."
            )

        self.num_capas = num_capas
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.campo_q = [
            np.zeros((self.num_rows, self.num_cols), dtype=np.complex128)
            for _ in range(self.num_capas)
        ]
        self.lock = threading.Lock()

        if propagation_coeffs and len(propagation_coeffs) != num_capas:
            raise ValueError(
                f"La lista 'propagation_coeffs' debe tener longitud {num_capas}"
            )
        self.propagation_coeffs = (
            propagation_coeffs
            if propagation_coeffs
            else [DEFAULT_PROPAGATION_COEFF] * num_capas
        )

        if dissipation_coeffs and len(dissipation_coeffs) != num_capas:
            raise ValueError(
                f"La lista 'dissipation_coeffs' debe tener longitud {num_capas}"
            )
        self.dissipation_coeffs = (
            dissipation_coeffs
            if dissipation_coeffs
            else [DEFAULT_DISSIPATION_COEFF] * num_capas
        )

        if not propagation_coeffs:
            logger.info(
                "Usando coeficiente de propagación por defecto para todas las capas."
            )
        if not dissipation_coeffs:
            logger.info(
                "Usando coeficiente de disipación por defecto para todas las capas."
            )

        logger.info(
            "Campo cimático toroidal inicializado: %d capas, dims=%dx%d",
            self.num_capas,
            self.num_rows,
            self.num_cols,
        )

    def aplicar_influencia(
        self, capa: int, row: int, col: int, vector: complex, nombre_watcher: str
    ) -> bool:
        """
        Aplica una influencia externa (perturbación de onda) a un punto
        específico del campo.

        Esto simula una perturbación localizada o una inyección de
        energía/amplitud en la grilla por parte de un watcher.

        Args:
            capa (int): Índice de la capa (0 a num_capas-1).
            row (int): Índice de la fila (0 a num_rows-1).
            col (int): Índice de la columna (0 a num_cols-1).
            vector (complex): Número complejo que representa la influencia (amplitud y fase).
            nombre_watcher (str): Nombre del watcher que aplica la influencia.

        Returns:
            bool: True si la influencia se aplicó correctamente, False en caso contrario.
        """
        if not (0 <= capa < self.num_capas):
            logger.error(
                "Error al aplicar influencia de '%s': índice de capa fuera "
                "de rango (%d). Rango válido: 0-%d.",
                nombre_watcher,
                capa,
                self.num_capas - 1,
            )
            return False
        if not (0 <= row < self.num_rows):
            logger.error(
                "Error al aplicar influencia de '%s': índice de fila fuera "
                "de rango (%d). Rango válido: 0-%d.",
                nombre_watcher,
                row,
                self.num_rows - 1,
            )
            return False
        if not (0 <= col < self.num_cols):
            logger.error(
                "Error al aplicar influencia de '%s': índice de columna "
                "fuera de rango (%d). Rango válido: 0-%d.",
                nombre_watcher,
                col,
                self.num_cols - 1,
            )
            return False
        if not isinstance(vector, complex):
            logger.error(
                "Error al aplicar influencia de '%s': vector de influencia "
                "inválido. Debe ser un número complejo. Recibido: %s",
                nombre_watcher,
                type(vector),
            )
            return False

        try:
            with self.lock:
                self.campo_q[capa][row, col] += vector
                valor_actual = self.campo_q[capa][row, col]
            logger.info(
                "'%s' aplicó influencia en capa %d, nodo (%d, %d): %s. Nuevo valor: %s",
                nombre_watcher,
                capa,
                row,
                col,
                vector,
                valor_actual,
            )
            return True
        except Exception:
            logger.exception(
                "Error inesperado al aplicar influencia de '%s' en (%d, %d, %d)",
                nombre_watcher,
                capa,
                row,
                col,
            )
            return False

    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Obtiene las coordenadas (row, col) de los 4 vecinos directos
        (arriba, abajo, izquierda, derecha) de un nodo, aplicando
        conectividad toroidal.
        """
        rows = np.array([row - 1, row + 1, row, row]) % self.num_rows
        cols = np.array([col, col, col - 1, col + 1]) % self.num_cols
        return list(zip(rows, cols, strict=False))

    def calcular_gradiente_adaptativo(self) -> np.ndarray:
        """
        Calcula el gradiente adaptativo (diferencia de magnitud) entre
        capas adyacentes. Puede interpretarse como una medida de la
        "tensión" o "interferencia" entre capas de ondas.
        """
        if self.num_capas < 2:
            logger.warning(
                "Se necesita al menos 2 capas para calcular el "
                "gradiente entre capas."
            )
            return np.array([])

        with self.lock:
            campo_copia = [np.copy(capa) for capa in self.campo_q]

        gradiente_entre_capas = np.zeros(
            (self.num_capas - 1, self.num_rows, self.num_cols)
        )
        for i in range(self.num_capas - 1):
            diferencia_vectorial = campo_copia[i] - campo_copia[i + 1]
            magnitud_diferencia = np.abs(diferencia_vectorial)
            gradiente_entre_capas[i] = magnitud_diferencia
        logger.debug(
            "Gradiente entre capas calculado "
            f"(shape: {gradiente_entre_capas.shape})"
        )
        return gradiente_entre_capas

    def get_energy_density_map(self) -> np.ndarray:
        """
        Genera un mapa de "densidad de energía", análogo a los patrones
        visibles en los experimentos de cimática.

        Este mapa representa la energía agregada de la onda (amplitud al cuadrado)
        en cada ubicación (fila, columna) a través de las capas. Las capas
        más internas (índices bajos) suelen tener mayor peso en la ponderación.

        Returns:
            np.ndarray: Array 2D (num_rows x num_cols) del mapa de densidad de energía.
        """
        pesos = (
            np.linspace(1.0, 0.5, self.num_capas)
            if self.num_capas > 1
            else np.array([1.0])
        )
        energy_density_map = np.zeros((self.num_rows, self.num_cols))
        with self.lock:
            campo_copia = [np.copy(capa) for capa in self.campo_q]

        for i, capa_actual in enumerate(campo_copia):
            # La energía es proporcional a la amplitud al cuadrado
            magnitud_capa = np.abs(capa_actual) ** 2
            energy_density_map += pesos[i] * magnitud_capa
        return energy_density_map

    def apply_wave_dynamics_step(self, dt: float, beta: float):
        """
        Simula la dinámica de la onda (advección, acoplamiento, disipación),
        análoga a la Ecuación de Onda en un medio.

        Esta dinámica simula:
        1. Advección/Propagación de la onda en la dirección azimutal (columnas),
           controlada por `propagation_coeffs` (por capa).
        2. Acoplamiento/Transporte vertical (entre filas), controlado por `beta`.
        3. Disipación de energía (decaimiento de amplitud), controlada por
           `dissipation_coeffs` (por capa).
        """
        if beta < 0:
            logger.warning(
                "El factor beta (acoplamiento vertical) debería ser no negativo."
            )

        with self.lock:
            next_campo = []
            # Pre-calcular arrays para broadcasting
            propagation_coeffs_array = np.array(self.propagation_coeffs)[
                :, np.newaxis, np.newaxis
            ]
            dissipation_coeffs_array = np.array(self.dissipation_coeffs)[
                :, np.newaxis, np.newaxis
            ]

            for capa_idx in range(self.num_capas):
                capa_actual = self.campo_q[capa_idx]

                # Vecindad con condiciones toroidales usando np.roll
                v_left = np.roll(capa_actual, shift=1, axis=1)
                v_up = np.roll(capa_actual, shift=1, axis=0)
                v_down = np.roll(capa_actual, shift=-1, axis=0)

                # Cálculo vectorizado para toda la capa
                damped = capa_actual * (1.0 - dissipation_coeffs_array[capa_idx] * dt)
                advected = propagation_coeffs_array[capa_idx] * v_left * dt
                coupled = beta * (v_up + v_down) * dt

                next_campo.append(damped + advected + coupled)

            self.campo_q = next_campo

    def set_uniform_potential_field(self, seed: Optional[int] = None):
        """
        Inicializa el campo a un estado de potencial uniforme (magnitud 1)
        con fases aleatorias.

        Asigna a cada nodo del campo un estado en el círculo unitario con una
        fase aleatoria, resultando en un número complejo `e^(i * random_angle)`.
        Esto representa un estado de energía potencial uniforme pero sin
        coherencia de fase.

        Args:
            seed (Optional[int]): Semilla para el generador de números
                                  aleatorios para reproducibilidad.
        """
        rng = np.random.default_rng(seed)
        with self.lock:
            for capa_idx in range(self.num_capas):
                random_angles = rng.uniform(
                    0, 2 * np.pi, size=(self.num_rows, self.num_cols)
                )
                self.campo_q[capa_idx] = np.exp(1j * random_angles)

    def apply_internal_phase_evolution(self, dt: float):
        """
        Simula la evolución intrínseca de la fase del medio.

        Este método simula la evolución de la fase de cada punto del campo
        según sus propiedades locales. La evolución sigue la ecuación de
        Schrödinger para un paso de tiempo `dt` con un Hamiltoniano simple.

        La transformación es: ψ(t+dt) = e^(-i * prop_coeff * dt) * ψ(t),
        donde `prop_coeff` es el coeficiente de propagación de la capa.

        Args:
            dt (float): El paso de tiempo para la evolución.
        """
        with self.lock:
            # Convertir propagation_coeffs a array NumPy para broadcasting
            propagation_coeffs_array = np.array(self.propagation_coeffs)
            # Calcular el cambio de fase para todas las capas a la vez
            phase_changes = np.exp(
                -1j * propagation_coeffs_array[:, np.newaxis, np.newaxis] * dt
            )

            # Aplicar el cambio de fase a todas las capas
            for i in range(self.num_capas):
                self.campo_q[i] *= phase_changes[i]


# --- Instancia Global y Lógica de Simulación ---
try:
    # Usar las constantes globales para la instancia del servicio
    campo_toroidal_global_servicio = ToroidalField(
        num_capas=NUM_CAPAS,
        num_rows=NUM_FILAS,
        num_cols=NUM_COLUMNAS,
        propagation_coeffs=None,  # Usará los defaults internos
        dissipation_coeffs=None,  # Usará los defaults internos
    )
    logger.info("Aplicando influencias iniciales al campo cimático global...")
    campo_toroidal_global_servicio.aplicar_influencia(
        capa=0, row=1, col=2, vector=complex(1.0, 0.5), nombre_watcher="watcher_init_1"
    )
    campo_toroidal_global_servicio.aplicar_influencia(
        capa=2, row=3, col=0, vector=complex(0.2, -0.1), nombre_watcher="watcher_init_2"
    )
    logger.info("Influencias iniciales aplicadas al campo cimático global.")

except ValueError:
    logger.exception("Error crítico al inicializar ToroidalField global. Terminando.")
    exit(1)
except Exception as e:
    logger.exception(f"Error inesperado durante la inicialización del servicio: {e}")
    exit(1)


# --- Hilo y Función de Simulación ---
simulation_thread = None
stop_simulation_event = threading.Event()


def cymatic_simulation_loop(dt: float, beta: float):
    """Bucle que ejecuta la simulación de la dinámica cimática periódicamente."""
    logger.info(f"Iniciando bucle de simulación cimática con dt={dt}, beta={beta}...")
    while not stop_simulation_event.is_set():
        try:
            start_time = time.monotonic()
            campo_toroidal_global_servicio.apply_wave_dynamics_step(dt, beta)
            campo_toroidal_global_servicio.apply_internal_phase_evolution(dt)
            elapsed = time.monotonic() - start_time
            sleep_time = max(0, dt - elapsed)
            stop_simulation_event.wait(sleep_time)
        except Exception as e:
            logger.error(
                f"Error en el bucle de simulación cimática: {e}", exc_info=True
            )
            stop_simulation_event.wait(5)


# --- Servidor Flask ---
app = Flask(__name__)


@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Verifica la salud del servicio de simulación cimática.

    Incluye el estado de inicialización del campo y el hilo de simulación.
    """
    sim_alive = simulation_thread.is_alive() if simulation_thread else False
    service_ready = campo_toroidal_global_servicio and hasattr(
        campo_toroidal_global_servicio, "num_capas"
    )
    status_code = 503
    response = {
        "status": "error",
        "message": "Servicio de simulación no completamente inicializado.",
        "simulation_running": sim_alive,
        "field_initialized": service_ready,
    }

    if service_ready and sim_alive:
        response["status"] = "success"
        response["message"] = "Servicio de simulación cimática saludable y activo."
        status_code = 200
    elif service_ready and not sim_alive:
        response["status"] = "warning"
        response["message"] = "Servicio inicializado pero la simulación no está activa."
        status_code = 503
    elif not service_ready:
        response["message"] = "Error: Objeto ToroidalField no inicializado."
        status_code = 500

    logger.debug(f"Health check: {response}")
    return jsonify(response), status_code


@app.route("/api/ecu", methods=["GET"])
def obtener_estado_unificado_api() -> Tuple[Any, int]:
    """
    Endpoint REST para obtener el mapa de densidad de energía del campo cimático.

    Retorna un array 2D (filas x columnas) que representa la energía
    agregada de la onda en cada punto, ponderada por capa.
    """
    try:
        energy_map = campo_toroidal_global_servicio.get_energy_density_map()
        response_data = {
            "status": "success",
            "energy_density_map": energy_map.tolist(),
            "metadata": {
                "descripcion": "Mapa de densidad de energía del campo cimático ponderado por capa",
                "capas": campo_toroidal_global_servicio.num_capas,
                "filas": campo_toroidal_global_servicio.num_rows,
                "columnas": campo_toroidal_global_servicio.num_cols,
            },
        }
        logger.info("Mapa de densidad de energía calculado y enviado.")
        return jsonify(response_data), 200
    except Exception as e:
        logger.exception("Error en endpoint /api/ecu: %s", e)
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Error interno del servidor al obtener el mapa de energía.",
                }
            ),
            500,
        )


def _validate_and_parse_influence_payload(
    data: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Any, int]]]:
    """Valida y parsea el payload de la solicitud de influencia."""
    if not data:
        return None, (
            jsonify({"status": "error", "message": "Payload JSON vacío o ausente"}),
            400,
        )

    required_fields = {
        "capa": int,
        "row": int,
        "col": int,
        "vector": list,
        "nombre_watcher": str,
    }
    missing = [f for f in required_fields if f not in data]
    if missing:
        msg = f"Faltan campos requeridos en el JSON: {', '.join(missing)}"
        return None, (jsonify({"status": "error", "message": msg}), 400)

    errors = []
    for field, expected_type in required_fields.items():
        if not isinstance(data[field], expected_type):
            errors.append(
                f"Campo '{field}' debe ser {expected_type.__name__}, "
                f"recibido {type(data[field]).__name__}"
            )

    vec_data = data.get("vector")
    vec_complex = None
    if isinstance(vec_data, list):
        if len(vec_data) != 2:
            errors.append("Campo 'vector' debe ser una lista de 2 elementos.")
        elif not all(isinstance(v, (int, float)) for v in vec_data):
            errors.append("Elementos del 'vector' deben ser números.")
        else:
            vec_complex = complex(vec_data[0], vec_data[1])

    if errors:
        msg = "Errores de tipo en JSON: " + "; ".join(errors).lower()
        return None, (jsonify({"status": "error", "message": msg}), 400)

    parsed = {
        "capa": data["capa"],
        "row": data["row"],
        "col": data["col"],
        "vector_complex": vec_complex,
        "nombre_watcher": data["nombre_watcher"],
    }

    if not (0 <= parsed["capa"] < campo_toroidal_global_servicio.num_capas):
        return None, (
            jsonify({"status": "error", "message": "Índice de capa fuera de rango."}),
            400,
        )
    if not (0 <= parsed["row"] < campo_toroidal_global_servicio.num_rows):
        return None, (
            jsonify({"status": "error", "message": "Índice de fila fuera de rango."}),
            400,
        )
    if not (0 <= parsed["col"] < campo_toroidal_global_servicio.num_cols):
        return None, (
            jsonify(
                {"status": "error", "message": "Índice de columna fuera de rango."}
            ),
            400,
        )

    return parsed, None


@app.route("/api/ecu/influence", methods=["POST"])
def recibir_influencia_malla() -> Tuple[Any, int]:
    """Recibe y aplica una influencia externa (perturbación) al campo cimático.

    Espera un payload JSON con 'capa', 'row', 'col', 'vector' ([real, imag]),
    y 'nombre_watcher'.
    """
    logger.info("Solicitud POST /api/ecu/influence recibida.")
    try:
        data = request.get_json()
        parsed_data, error_response = _validate_and_parse_influence_payload(data)

        if error_response:
            return error_response

        success = campo_toroidal_global_servicio.aplicar_influencia(
            capa=parsed_data["capa"],
            row=parsed_data["row"],
            col=parsed_data["col"],
            vector=parsed_data["vector_complex"],
            nombre_watcher=parsed_data["nombre_watcher"],
        )

        if success:
            logger.info(
                "Influencia de '%s' aplicada exitosamente via API en (%d, %d, %d).",
                parsed_data["nombre_watcher"],
                parsed_data["capa"],
                parsed_data["row"],
                parsed_data["col"],
            )
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": (
                            f"Influencia de '{parsed_data['nombre_watcher']}' "
                            "aplicada."
                        ),
                        "applied_to": {
                            "capa": parsed_data["capa"],
                            "row": parsed_data["row"],
                            "col": parsed_data["col"],
                        },
                        "vector": [
                            parsed_data["vector_complex"].real,
                            parsed_data["vector_complex"].imag,
                        ],
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Error de validación interno al aplicar influencia.",
                    }
                ),
                400,
            )

    except Exception as e:
        logger.exception("Error inesperado en endpoint /api/ecu/influence: %s", e)
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Error interno al procesar la influencia.",
                }
            ),
            500,
        )


@app.route("/api/ecu/field_vector", methods=["GET"])
def get_field_vector_api() -> Tuple[Any, int]:
    """
    Endpoint REST para obtener el campo de ondas completo (amplitud y fase).
    Retorna una estructura 3D donde cada elemento es [real, imag].
    Shape: [num_capas, num_rows, num_cols, 2].
    """
    try:
        with campo_toroidal_global_servicio.lock:
            campo_copia = [
                [[[cell.real, cell.imag] for cell in row] for row in layer]
                for layer in campo_toroidal_global_servicio.campo_q
            ]

        logger.info(
            "Solicitud GET /api/ecu/field_vector recibida. "
            "Devolviendo campo de ondas completo."
        )
        response_data = {
            "status": "success",
            "field_vector": campo_copia,
            "metadata": {
                "descripcion": "Campo de ondas complejo en grilla 3D toroidal",
                "capas": campo_toroidal_global_servicio.num_capas,
                "filas": campo_toroidal_global_servicio.num_rows,
                "columnas": campo_toroidal_global_servicio.num_cols,
            },
        }
        return jsonify(response_data), 200
    except Exception as e_field_vector:
        logger.exception(f"Error en endpoint /api/ecu/field_vector: {e_field_vector}")
        error_response = {
            "status": "error",
            "message": "Error interno al procesar la solicitud del campo de ondas.",
        }
        return jsonify(error_response), 500


@app.route("/debug/set_random_phase", methods=["POST"])
def set_random_phase_endpoint():
    """
    Endpoint de depuración para reiniciar el campo a un estado de potencial
    uniforme con fase aleatoria.
    Protegido para ejecutarse solo en entornos de no producción.
    """
    flask_env = os.environ.get("FLASK_ENV", "production").lower().strip()
    logger.debug(
        f"Verificando entorno para endpoint de depuración. FLASK_ENV='{flask_env}'"
    )

    if flask_env not in ["development", "test"]:
        logger.warning(
            "Acceso denegado a endpoint de depuración. Entorno actual: '%s'.",
            flask_env,
        )
        return (
            jsonify(
                {
                    "status": "error",
                    "message": (
                        "Endpoint de depuración no disponible en entorno "
                        f"'{flask_env}'."
                    ),
                }
            ),
            403,
        )

    try:
        campo_toroidal_global_servicio.set_uniform_potential_field()
        logger.info(
            "Campo reiniciado a potencial uniforme con fase aleatoria "
            "a través de endpoint de depuración."
        )
        return jsonify({"status": "success", "message": "Campo reiniciado"})
    except Exception as e:
        logger.exception(f"Error en set_random_phase: {e}")
        return (
            jsonify(
                {"status": "error", "message": "Error interno al reiniciar el campo."}
            ),
            500,
        )


@app.route("/api/ecu/field_vector/region/<int:capa_idx>", methods=["GET"])
def get_region_field_vector_api(capa_idx: int) -> Tuple[Any, int]:
    """
    Endpoint REST para obtener el campo de ondas de una capa específica.
    """
    if not (0 <= capa_idx < campo_toroidal_global_servicio.num_capas):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": (
                        "Índice de capa fuera de rango. Se esperaba entre 0 y "
                        f"{campo_toroidal_global_servicio.num_capas - 1}."
                    ),
                }
            ),
            404,
        )

    try:
        with campo_toroidal_global_servicio.lock:
            layer_copy = [
                [[cell.real, cell.imag] for cell in row]
                for row in campo_toroidal_global_servicio.campo_q[capa_idx]
            ]

        logger.info(f"Solicitud GET para la capa de ondas {capa_idx} recibida.")
        response_data = {
            "status": "success",
            "field_vector_region": layer_copy,
            "metadata": {
                "descripcion": f"Campo de ondas complejo para la capa {capa_idx}",
                "capa_idx": capa_idx,
                "filas": campo_toroidal_global_servicio.num_rows,
                "columnas": campo_toroidal_global_servicio.num_cols,
            },
        }
        return jsonify(response_data), 200
    except Exception as e:
        logger.exception(f"Error en endpoint para la capa {capa_idx}: {e}")
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Error interno al procesar la solicitud de la región.",
                }
            ),
            500,
        )


# --- Función Principal y Arranque ---
def main():
    """Función principal que configura el logging y arranca la aplicación."""
    global simulation_thread

    if not logging.getLogger("matriz_ecu").hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger.info(
        f"Configuración de simulación cimática: {NUM_CAPAS}x{NUM_FILAS}x{NUM_COLUMNAS}, "
        f"SimInterval={SIMULATION_INTERVAL}s, Beta={BETA_COUPLING}"
    )

    logger.info("Creando e iniciando hilo de simulación cimática...")
    stop_simulation_event.clear()
    simulation_thread = threading.Thread(
        target=cymatic_simulation_loop,
        args=(SIMULATION_INTERVAL, BETA_COUPLING),
        daemon=True,
        name="CymaticSimLoop",
    )
    simulation_thread.start()

    puerto_servicio = int(os.environ.get("MATRIZ_ECU_PORT", 8000))
    logger.info(
        f"Iniciando servicio Flask de simulación cimática en puerto {puerto_servicio}..."
    )
    app.run(host="0.0.0.0", port=puerto_servicio, debug=False, use_reloader=False)


# --- Punto de Entrada ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado detectada.")
    finally:
        logger.info("Enviando señal de detención al hilo de simulación cimática...")
        stop_simulation_event.set()
        if simulation_thread and simulation_thread.is_alive():
            logger.info("Esperando finalización del hilo de simulación...")
            simulation_thread.join(timeout=SIMULATION_INTERVAL * 2)
            if simulation_thread.is_alive():
                logger.warning(
                    "El hilo de simulación cimática no terminó limpiamente."
                )
        logger.info("Servicio de simulación cimática finalizado.")
