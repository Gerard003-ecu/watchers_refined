#!/usr/bin/env python3
"""Módulo que define la matriz ECU (Experiencia de Campo Unificado).

Este módulo modela un campo de confinamiento magnético toroidal (tipo Tokamak)
y la dinámica de los watchers dentro de él.

El campo es un campo vectorial 2D en una grilla 3D toroidal, donde cada punto
almacena un vector [vx, vy] que representa componentes de la densidad de flujo
magnético (B).
La interpretación exacta de vx y vy en términos de direcciones físicas
(toroidal, poloidal, radial) se puede definir conceptualmente por analogía.
Por ejemplo:
vx es la componente toroidal de B.
vy es la componente poloidal (vertical) de B.
La dimensión de la "capa" podría representar la dirección radial.

Proporciona una API REST para:
- Obtener el estado unificado del campo.
- Recibir influencias externas (perturbaciones).
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
# El logger se configura dentro de main() para ser compatible con contenedores.
# A nivel global, solo obtenemos la instancia del logger.
# Los logs emitidos antes de que main() configure el logger (ej. warnings de
# get_env_*) usarán la configuración por defecto de logging (a stderr).
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


# --- Constantes Configurables para el Campo Toroidal ---
NUM_CAPAS = get_env_int("ECU_NUM_CAPAS", 3)
NUM_FILAS = get_env_int("ECU_NUM_FILAS", 4)
NUM_COLUMNAS = get_env_int("ECU_NUM_COLUMNAS", 5)
DEFAULT_ALPHA_VALUE = get_env_float("ECU_DEFAULT_ALPHA", 0.5)
DEFAULT_DAMPING_VALUE = get_env_float("ECU_DEFAULT_DAMPING", 0.05)
SIMULATION_INTERVAL = get_env_float("ECU_SIM_INTERVAL", 1.0)
BETA_COUPLING = get_env_float("ECU_BETA_COUPLING", 0.1)


class ToroidalField:
    """
    Representa un campo de confinamiento magnético toroidal discreto.

    El campo se modela en una grilla 3D (capas x filas x columnas).
    Cada punto de la grilla (capa, fila, columna) almacena un vector 2D
    [vx, vy], interpretado como componentes locales de la densidad de
    flujo magnético (B).
    La dinámica simula la evolución de este campo bajo efectos de
    advección/rotación, acoplamiento vertical y disipación, inspirada
    en la física de plasmas confinados.
    """

    def __init__(
        self,
        num_capas: int,
        num_rows: int,
        num_cols: int,
        alphas: Optional[List[float]] = None,
        dampings: Optional[List[float]] = None,
    ):
        """
        Inicializa el campo toroidal con dimensiones y parámetros por capa.

        Args:
            num_capas (int): Número de capas (dimensión radial/profundidad).
            num_rows (int): Número de filas (dimensión poloidal/vertical).
            num_cols (int): Número de columnas (dimensión toroidal/azimutal).
            alphas (Optional[List[float]]): Coeficientes de
                                            advección/rotación por capa.
                                            Relacionado con la dinámica
                                            toroidal del campo.
            dampings (Optional[List[float]]): Coeficientes de
                                              disipación/amortiguación
                                              por capa. Relacionado con la
                                              pérdida de intensidad del
                                              campo local.
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

        if alphas and len(alphas) != num_capas:
            raise ValueError(f"La lista 'alphas' debe tener longitud {num_capas}")
        self.alphas = alphas if alphas else [DEFAULT_ALPHA_VALUE] * num_capas

        if dampings and len(dampings) != num_capas:
            raise ValueError(f"La lista 'dampings' debe tener longitud {num_capas}")
        self.dampings = dampings if dampings else [DEFAULT_DAMPING_VALUE] * num_capas

        if not alphas:
            logger.info("Usando alpha por defecto para todas las capas.")
        if not dampings:
            logger.info("Usando damping por defecto para todas las capas.")

        logger.info(
            "Campo toroidal inicializado: %d capas, dims=%dx%d",
            self.num_capas,
            self.num_rows,
            self.num_cols,
        )

    def aplicar_influencia(
        self, capa: int, row: int, col: int, vector: complex, nombre_watcher: str
    ) -> bool:
        """
        Aplica una influencia externa (vector 2D) a un punto específico
        del campo.

        Esto simula una perturbación localizada o una inyección de
        energía/flujo en la grilla toroidal por parte de un watcher.

        Args:
            capa (int): Índice de la capa (0 a num_capas-1).
            row (int): Índice de la fila (0 a num_rows-1).
            col (int): Índice de la columna (0 a num_cols-1).
            vector (complex): Número complejo que representa la influencia.
            nombre_watcher (str): Nombre del watcher que aplica la influencia.

        Returns:
            bool: True si la influencia se aplicó correctamente,
                  False en caso contrario.
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
                "Error inesperado al aplicar influenciade '%s' en (%d, %d, %d)",
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
        # Versión NumPy-like, aunque para 4 vecinos la diferencia es mínima.
        # Muestra un estilo de código más vectorizado.
        rows = np.array([row - 1, row + 1, row, row]) % self.num_rows
        cols = np.array([col, col, col - 1, col + 1]) % self.num_cols
        return list(zip(rows, cols, strict=False))

    def calcular_gradiente_adaptativo(self) -> np.ndarray:
        """
        Calcula el gradiente adaptativo (diferencia de magnitud) entre
        capas de confinamiento adyacentes. Puede interpretarse como una
        medida de la "tensión" o "shear" del campo entre capas.
        """
        if self.num_capas < 2:
            logger.warning(
                "Se necesita al menos 2 capas para calcular el "
                "gradiente de confinamiento."
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
            "Gradiente de confinamiento entre capas calculado "
            f"(shape: {gradiente_entre_capas.shape})"
        )
        return gradiente_entre_capas

    def obtener_campo_unificado(self) -> np.ndarray:
        """
        Genera un mapa escalar unificado (num_rows x num_cols) ponderado
        por capa.

        Este mapa representa la intensidad o magnitud agregada del campo
        vectorial en cada ubicación (fila, columna) a través de las capas.
        Puede interpretarse como una medida de la "densidad de energía" o
        "turbulencia" local del campo. Las capas más internas (índices bajos)
        suelen tener mayor peso.

        Returns:
            np.ndarray: Array 2D (num_rows x num_cols) del campo unificado.
        """
        pesos = (
            np.linspace(1.0, 0.5, self.num_capas)
            if self.num_capas > 1
            else np.array([1.0])
        )
        campo_unificado = np.zeros((self.num_rows, self.num_cols))
        with self.lock:
            campo_copia = [np.copy(capa) for capa in self.campo_q]

        for i, capa_actual in enumerate(campo_copia):
            magnitud_capa = np.abs(capa_actual)
            campo_unificado += pesos[i] * magnitud_capa
        return campo_unificado

    def apply_rotational_step(self, dt: float, beta: float):
        """
        Aplica un paso de simulación (método numérico discreto) para la
        dinámica del campo vectorial toroidal de forma vectorizada.

        Esta dinámica simula:
        1. Advección/Rotación en la dirección toroidal (columnas),
           controlada por `alpha` (por capa).
        2. Acoplamiento/Transporte vertical (entre filas),
           controlado por `beta`.
        3. Disipación/Decaimiento local, controlada por `damping` (por capa).
        """
        if beta < 0:
            logger.warning(
                "El factor beta (acoplamiento vertical) debería ser no negativo."
            )

        with self.lock:
            next_campo = []
            # Pre-calcular arrays de alphas y dampings para broadcasting
            alphas_array = np.array(self.alphas)[:, np.newaxis, np.newaxis]
            dampings_array = np.array(self.dampings)[:, np.newaxis, np.newaxis]

            for capa_idx in range(self.num_capas):
                capa_actual = self.campo_q[capa_idx]

                # Vecindad con condiciones toroidales usando np.roll
                v_left = np.roll(capa_actual, shift=1, axis=1)
                v_up = np.roll(capa_actual, shift=1, axis=0)
                v_down = np.roll(capa_actual, shift=-1, axis=0)

                # Cálculo vectorizado para toda la capa
                damped = capa_actual * (1.0 - dampings_array[capa_idx] * dt)
                advected = alphas_array[capa_idx] * v_left * dt
                coupled = beta * (v_up + v_down) * dt

                next_campo.append(damped + advected + coupled)

            self.campo_q = next_campo

    def set_initial_quantum_phase(self, seed: Optional[int] = None):
        """
        Inicializa la fase cuántica del campo a un estado aleatorio.

        Asigna a cada nodo (c, r, c) un estado cuántico inicial en el
        círculo unitario con una fase aleatoria, resultando en un
        número complejo `e^(i * random_angle)`.

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

    def apply_quantum_step(self, dt: float):
        """
        Aplica un paso de evolución cuántica discreta al campo de forma
        vectorizada.

        Este método simula la evolución de la fase de cada estado cuántico
        en la grilla bajo un Hamiltoniano simple. La evolución sigue la
        ecuación de Schrödinger para un paso de tiempo `dt`.

        La transformación es: |ψ(t+dt)> = e^(-i * α * dt) * |ψ(t)>,
        donde α es un coeficiente específico de la capa.

        Args:
            dt (float): El paso de tiempo para la evolución.
        """
        with self.lock:
            # Convertir alphas a array NumPy para broadcasting
            alphas_array = np.array(self.alphas)  # Shape (num_capas,)
            # Calcular el cambio de fase para todas las capas a la vez
            # np.newaxis agrega dimensiones para que se broadcastee
            # con las dimensiones (rows, cols)
            phase_changes = np.exp(-1j * alphas_array[:, np.newaxis, np.newaxis] * dt)

            # Aplicar el cambio de fase a todas las capas
            # Esto funciona porque campo_q es una lista de arrays 2D
            for i in range(self.num_capas):
                self.campo_q[i] *= phase_changes[i]


# --- Instancia Global y Lógica de Simulación ---
# La instancia `campo_toroidal_global` y el evento `stop_event`
# fueron eliminados por ser redundantes o no utilizados.
try:
    # Usar las constantes globales para la instancia del servicio
    campo_toroidal_global_servicio = ToroidalField(
        num_capas=NUM_CAPAS,
        num_rows=NUM_FILAS,
        num_cols=NUM_COLUMNAS,
        alphas=None,  # Usará los defaults internos basados en num_capas
        dampings=None,  # Usará los defaults internos basados en num_capas
    )
    logger.info(
        "Aplicando influencias iniciales al campo de confinamiento global (servicio)..."
    )
    campo_toroidal_global_servicio.aplicar_influencia(
        capa=0, row=1, col=2, vector=complex(1.0, 0.5), nombre_watcher="watcher_init_1"
    )
    campo_toroidal_global_servicio.aplicar_influencia(
        capa=2, row=3, col=0, vector=complex(0.2, -0.1), nombre_watcher="watcher_init_2"
    )
    logger.info("Influencias iniciales aplicadas al campo global (servicio).")

except ValueError:
    logger.exception(
        "Error crítico al inicializar ToroidalField global (servicio). Terminando."
    )
    exit(1)
except Exception as e:
    logger.exception(f"Error inesperado durante la inicialización del servicio: {e}")
    exit(1)


# --- Hilo y Función de Simulación ---
simulation_thread = None
stop_simulation_event = threading.Event()


def simulation_loop(dt: float, beta: float):
    """Bucle que ejecuta la simulación dinámica periódicamente."""
    logger.info(f"Iniciando bucle de simulación ECU con dt={dt}, beta={beta}...")
    # Removed duplicated inner function definition.
    # The loop now correctly belongs to the outer simulation_loop function.
    # Also corrected stop_event to
    # stop_simulation_event and campo_toroidal_global
    # to campo_toroidal_global_servicio for
    # consistency with the rest of the module.
    while not stop_simulation_event.is_set():
        try:
            start_time = time.monotonic()
            campo_toroidal_global_servicio.apply_rotational_step(dt, beta)
            campo_toroidal_global_servicio.apply_quantum_step(dt)
            elapsed = time.monotonic() - start_time
            sleep_time = max(0, dt - elapsed)
            stop_simulation_event.wait(sleep_time)
        except Exception as e:
            logger.error(f"Error en el bucle de simulación: {e}", exc_info=True)
            # Opcional: esperar un poco antes de reintentar para
            # evitar un bucle de error rápido
            stop_simulation_event.wait(5)


# --- Servidor Flask ---
app = Flask(__name__)


# ... (Endpoints /api/health, /api/ecu, /api/ecu/influence
# sin cambios funcionales) ...
@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Verifica la salud del servicio ECU (Experiencia de Campo Unificado).

    Incluye el estado de inicialización del campo toroidal y el hilo
    de simulación.
    """
    sim_alive = simulation_thread.is_alive() if simulation_thread else False
    service_ready = campo_toroidal_global_servicio and hasattr(
        campo_toroidal_global_servicio, "num_capas"
    )
    status_code = 503  # Service Unavailable por defecto
    response = {
        "status": "error",
        "message": "Servicio ECU no completamente inicializado.",
        "simulation_running": sim_alive,
        "field_initialized": service_ready,
    }

    if service_ready and sim_alive:
        response["status"] = "success"
        response["message"] = "Servicio ECU saludable y simulación activa."
        status_code = 200
    elif service_ready and not sim_alive:
        response["status"] = "warning"
        response["message"] = (
            "Servicio ECU inicializado pero la simulación no está activa."
        )
        status_code = 503  # O 200 con warning, depende de la criticidad
    elif not service_ready:
        response["message"] = "Error: Objeto ToroidalField no inicializado."
        status_code = 500  # Internal server error

    logger.debug(f"Health check: {response}")
    return jsonify(response), status_code


@app.route("/api/ecu", methods=["GET"])
def obtener_estado_unificado_api() -> Tuple[Any, int]:
    """
    Endpoint REST para obtener el mapa escalar unificado del campo toroidal.

    Retorna un array 2D (filas x columnas) que representa la intensidad
    agregada del campo vectorial en cada punto, ponderada por capa.
    """
    try:
        campo_unificado = campo_toroidal_global_servicio.obtener_campo_unificado()
        # SOLUCIÓN E501: Se formatea el diccionario para mayor legibilidad.
        response_data = {
            "status": "success",
            "estado_campo_unificado": campo_unificado.tolist(),
            "metadata": {
                "descripcion": "Mapa de intensidad del campo toroidal ponderado por capa",
                "capas": campo_toroidal_global_servicio.num_capas,
                "filas": campo_toroidal_global_servicio.num_rows,
                "columnas": campo_toroidal_global_servicio.num_cols,
            },
        }
        return jsonify(response_data), 200
    except Exception as e:
        logger.exception("Error en endpoint /api/ecu: %s", e)
        return jsonify(
            {
                "status": "error",
                "message": "Error interno del servidor al obtener estado unificado.",
            }
        ), 500


def _validate_and_parse_influence_payload(
    data: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Any, int]]]:
    """Valida y parsea el payload de la solicitud de influencia.

    Args:
        data: El diccionario JSON de la solicitud.

    Returns:
        Una tupla `(parsed_data, error_response)`.
        Si la validación es exitosa, `parsed_data` contiene los datos
        validados y `error_response` es `None`.
        Si la validación falla, `parsed_data` es `None` y `error_response`
        contiene una tupla `(response_json, status_code)` para ser
        retornada por el endpoint.
    """
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
    """Recibe y aplica una influencia externa al campo toroidal.

    Espera un payload JSON con 'capa', 'row', 'col', 'vector' (lista 2D),
    y 'nombre_watcher'.

    Returns:
        Una tupla de respuesta JSON y código de estado HTTP.
        - 200 OK: Si la influencia se aplicó con éxito.
        - 400 Bad Request: Si el payload es inválido o faltan campos.
        - 500 Internal Server Error: Si ocurre un error inesperado.
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
            return jsonify(
                {
                    "status": "success",
                    "message": f"Influencia de '{parsed_data['nombre_watcher']}' aplicada.",
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
            ), 200
        else:
            # Este caso puede ser redundante si _validate_... es exhaustivo
            return jsonify(
                {
                    "status": "error",
                    "message": "Error de validación interno al aplicar influencia.",
                }
            ), 400

    except Exception as e:
        logger.exception("Error inesperado en endpoint /api/ecu/influence: %s", e)
        return jsonify(
            {"status": "error", "message": "Error interno al procesar la influencia."}
        ), 500


# Retorna el campo vectorial completo
@app.route("/api/ecu/field_vector", methods=["GET"])
def get_field_vector_api() -> Tuple[Any, int]:
    """
    Endpoint REST para obtener
    el campo vectorial completo de la grilla toroidal.
    Retorna una estructura 3D (lista de listas de listas)
    donde cada elemento es un vector 2D [vx, vy].
    Shape: [num_capas, num_rows, num_cols, 2].
    """
    try:
        with campo_toroidal_global_servicio.lock:
            # Obtener una copia del campo para evitar modificarlo mientras
            # se serializa
            campo_copia = [
                [[[cell.real, cell.imag] for cell in row] for row in layer]
                for layer in campo_toroidal_global_servicio.campo_q
            ]

        logger.info(
            "Solicitud GET /api/ecu/field_vector recibida. "
            "Devolviendo campo vectorial completo."
        )
        response_data = {
            "status": "success",
            "field_vector": campo_copia,
            "metadata": {
                "descripcion": (
                    "Campo vectorial 2D en grilla 3D toroidal "
                    "(Analogía Densidad Flujo Magnético)"
                ),
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
            "message": "Error interno al procesar la solicitud del campo vectorial.",
        }
        return jsonify(error_response), 500


@app.route("/debug/set_random_phase", methods=["POST"])
def set_random_phase_endpoint():
    """
    Endpoint de depuración para reiniciar el campo a un estado de fase aleatoria.
    Protegido para ejecutarse solo en entornos de no producción.
    """
    # Leer la variable de entorno y loguear su valor para depuración
    flask_env = os.environ.get("FLASK_ENV", "production").lower().strip()
    logger.debug(
        f"Verificando entorno para endpoint de depuración. FLASK_ENV='{flask_env}'"
    )

    # Comprobar si el entorno permite la ejecución de este endpoint
    if flask_env not in ["development", "test"]:
        logger.warning(
            f"Acceso denegado a endpoint de depuración. Entorno actual: '{flask_env}'."
        )
        return jsonify(
            {
                "status": "error",
                "message": f"Endpoint de depuración no disponible en entorno '{flask_env}'.",
            }
        ), 403

    try:
        campo_toroidal_global_servicio.set_initial_quantum_phase()
        logger.info(
            "Campo reiniciado a fase cuántica aleatoria a través de endpoint de depuración."
        )
        return jsonify({"status": "success", "message": "Campo reiniciado"})
    except Exception as e:
        logger.exception(f"Error en set_random_phase: {e}")
        return jsonify(
            {"status": "error", "message": "Error interno al reiniciar el campo."}
        ), 500


@app.route("/api/ecu/field_vector/region/<int:capa_idx>", methods=["GET"])
def get_region_field_vector_api(capa_idx: int) -> Tuple[Any, int]:
    """
    Endpoint REST para obtener el campo vectorial de una capa específica.
    """
    if not (0 <= capa_idx < campo_toroidal_global_servicio.num_capas):
        return jsonify(
            {
                "status": "error",
                "message": f"Índice de capa fuera de rango. Se esperaba entre 0 y {campo_toroidal_global_servicio.num_capas - 1}.",
            }
        ), 404

    try:
        with campo_toroidal_global_servicio.lock:
            # Obtener una copia de la capa para evitar problemas de concurrencia
            layer_copy = [
                [[cell.real, cell.imag] for cell in row]
                for row in campo_toroidal_global_servicio.campo_q[capa_idx]
            ]

        logger.info(f"Solicitud GET para la capa {capa_idx} recibida.")
        response_data = {
            "status": "success",
            "field_vector_region": layer_copy,
            "metadata": {
                "descripcion": f"Campo vectorial 2D para la capa {capa_idx}",
                "capa_idx": capa_idx,
                "filas": campo_toroidal_global_servicio.num_rows,
                "columnas": campo_toroidal_global_servicio.num_cols,
            },
        }
        return jsonify(response_data), 200
    except Exception as e:
        logger.exception(f"Error en endpoint para la capa {capa_idx}: {e}")
        return jsonify(
            {
                "status": "error",
                "message": "Error interno al procesar la solicitud de la región.",
            }
        ), 500


# --- Función Principal y Arranque ---
def main():
    """Función principal que configura el logging y arranca la aplicación."""
    global simulation_thread

    # --- Configuración del Logging ---
    # Refactorizado para loguear siempre a stdout, siguiendo 12-Factor App.
    # Se elimina la gestión de archivos de log y directorios.
    if not logging.getLogger("matriz_ecu").hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format=(
                "%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s"
            ),
            handlers=[
                logging.StreamHandler(sys.stdout)  # Redirige los logs a la consola
            ],
        )
    # --- Fin Configuración del Logging ---

    logger.info(
        f"Configuración ECU: {NUM_CAPAS}x{NUM_FILAS}x{NUM_COLUMNAS}, "
        f"SimInterval={SIMULATION_INTERVAL}s, Beta={BETA_COUPLING}"
    )

    logger.info("Creando e iniciando hilo de simulación ECU...")
    stop_simulation_event.clear()
    simulation_thread = threading.Thread(
        target=simulation_loop,
        args=(SIMULATION_INTERVAL, BETA_COUPLING),
        daemon=True,
        name="ECUSimLoop",
    )
    simulation_thread.start()

    puerto_servicio = int(os.environ.get("MATRIZ_ECU_PORT", 8000))
    logger.info(f"Iniciando servicio Flask de ecu en puerto {puerto_servicio}...")
    app.run(host="0.0.0.0", port=puerto_servicio, debug=False, use_reloader=False)


# --- Punto de Entrada ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado detectada.")
    finally:
        logger.info("Enviando señal de detención al hilo de simulación ECU...")
        stop_simulation_event.set()
        if simulation_thread and simulation_thread.is_alive():
            logger.info("Esperando finalización del hilo de simulación...")
            simulation_thread.join(timeout=SIMULATION_INTERVAL * 2)
            if simulation_thread.is_alive():
                logger.warning("El hilo de simulación ECU no terminó limpiamente.")
        logger.info("Servicio matriz_ecu finalizado.")
