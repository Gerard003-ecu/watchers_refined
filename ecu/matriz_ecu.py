# --- START OF FILE matriz_ecu.py (CORREGIDO) ---

# !/usr/bin/env python3
"""
Módulo que define la matriz ECU (Experiencia de Campo Unificado)
para el sistema Watchers, inspirada en un campo de confinamiento magnético
toroidal (tipo Tokamak). Modela el estado y la dinámica de los watchers
dentro de este campo multicapa.
Proporciona servicios REST para obtener el estado unificado y recibir
influencias. Incluye un hilo de simulación en segundo plano para la
evolución dinámica del campo.

El campo es representado como un campo vectorial 2D discreto en una
grilla 3D toroidal. Cada punto de la grilla (capa, fila, columna)
almacena un vector [vx, vy], que puede interpretarse como componentes
locales de la densidad de flujo magnético (B).
La interpretación exacta de vx y vy en términos de direcciones físicas
(toroidal, poloidal, radial) se puede definir conceptualmente por analogía.
Por ejemplo:
vx es la componente toroidal de B.
vy es la componente poloidal (vertical) de B.
La dimensión de la "capa" podría representar la dirección radial.
"""

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request

# --- Configuración del Logging ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
if not logging.getLogger("matriz_ecu").hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        # Añadido threadName
        format=(
            "%(asctime)s [%(levelname)s] [%(threadName)s] "
            "%(name)s: %(message)s"
        ),
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "matriz_ecu.log")),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger("matriz_ecu")

# --- Constantes Configurables para el Campo Toroidal ---
# --- Constantes Configurables ---
NUM_CAPAS = int(os.environ.get("ECU_NUM_CAPAS", 3))
NUM_FILAS = int(os.environ.get("ECU_NUM_FILAS", 4))
NUM_COLUMNAS = int(os.environ.get("ECU_NUM_COLUMNAS", 5))
DEFAULT_ALPHA_VALUE = 0.5
DEFAULT_DAMPING_VALUE = 0.05
SIMULATION_INTERVAL = float(os.environ.get("ECU_SIM_INTERVAL", 1.0))
BETA_COUPLING = float(os.environ.get("ECU_BETA_COUPLING", 0.1))


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
            self.num_capas, self.num_rows, self.num_cols
        )

    def aplicar_influencia(
        self, capa: int, row: int, col: int, vector: complex,
        nombre_watcher: str
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
                nombre_watcher, capa, self.num_capas - 1
            )
            return False
        if not (0 <= row < self.num_rows):
            logger.error(
                "Error al aplicar influencia de '%s': índice de fila fuera "
                "de rango (%d). Rango válido: 0-%d.",
                nombre_watcher, row, self.num_rows - 1
            )
            return False
        if not (0 <= col < self.num_cols):
            logger.error(
                "Error al aplicar influencia de '%s': índice de columna "
                "fuera de rango (%d). Rango válido: 0-%d.",
                nombre_watcher, col, self.num_cols - 1
            )
            return False
        if not isinstance(vector, complex):
            logger.error(
                "Error al aplicar influencia de '%s': vector de influencia "
                "inválido. Debe ser un número complejo. Recibido: %s",
                nombre_watcher, type(vector)
            )
            return False

        try:
            with self.lock:
                self.campo_q[capa][row, col] += vector
                valor_actual = self.campo_q[capa][row, col]
            logger.info(
                "'%s' aplicó influencia en capa %d, nodo (%d, %d): %s. "
                "Nuevo valor: %s",
                nombre_watcher,
                capa,
                row,
                col,
                vector,
                valor_actual)
            return True
        except Exception:
            logger.exception(
                "Error inesperado al aplicar influencia"
                "de '%s' en (%d, %d, %d)",
                nombre_watcher, capa, row, col
            )
            return False

    def get_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """
        Obtiene las coordenadas (row, col) de los 4 vecinos directos
        (arriba, abajo, izquierda, derecha) de un nodo, aplicando
        conectividad toroidal en la dimensión de columna.
        """
        neighbors = []
        up_row = (row - 1) % self.num_rows
        neighbors.append((up_row, col))
        down_row = (row + 1) % self.num_rows
        neighbors.append((down_row, col))
        left_col = (col - 1) % self.num_cols
        neighbors.append((row, left_col))
        right_col = (col + 1) % self.num_cols
        neighbors.append((row, right_col))
        return neighbors

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

        gradiente_entre_capas = np.zeros((self.num_capas - 1,
                                          self.num_rows, self.num_cols))
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
            np.linspace(1.0, 0.5, self.num_capas) if self.num_capas > 1
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
        dinámica del campo vectorial toroidal.

        Esta dinámica simula:
        1. Advección/Rotación en la dirección toroidal (columnas),
           controlada por `alpha` (por capa).
        2. Acoplamiento/Transporte vertical (entre filas),
           controlado por `beta`.
        3. Disipación/Decaimiento local, controlada por `damping` (por capa).

        Aproxima la evolución del campo vectorial V según una ecuación tipo:
        dV/dt ≈ - alpha * dV/d(toroidal_angle) + beta_up * V_up +
                beta_down * V_down - damping * V
        (Esta es una simplificación conceptual de la PDE subyacente).

        Args:
            dt (float): Paso de tiempo de la simulación.
            beta (float): Coeficiente de acoplamiento vertical global.
        """
        if beta < 0:
            logger.warning("El factor beta (acoplamiento vertical) "
                           "debería ser no negativo.")

        with self.lock:
            next_campo = [np.copy(capa) for capa in self.campo_q]
            for capa_idx in range(self.num_capas):
                alpha_capa = self.alphas[capa_idx]
                damping_capa = self.dampings[capa_idx]
                for r in range(self.num_rows):
                    for c in range(self.num_cols):
                        v_curr = self.campo_q[capa_idx][r, c]
                        v_left = self.campo_q[capa_idx][
                            r, (c - 1) % self.num_cols
                        ]
                        v_up = self.campo_q[capa_idx][
                            (r - 1) % self.num_rows, c
                        ]
                        v_down = self.campo_q[capa_idx][
                            (r + 1) % self.num_rows, c
                        ]

                        damped = v_curr * (1.0 - damping_capa * dt)
                        advected = alpha_capa * v_left * dt
                        coupled = beta * (v_up + v_down) * dt

                        next_campo[capa_idx][r, c] = (
                            damped + advected + coupled
                        )
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
        Aplica un paso de evolución cuántica discreta al campo.

        Este método simula la evolución de la fase de cada estado cuántico
        en la grilla bajo un Hamiltoniano simple. La evolución sigue la
        ecuación de Schrödinger para un paso de tiempo `dt`.

        La transformación es: |ψ(t+dt)> = e^(-i * α * dt) * |ψ(t)>,
        donde α es un coeficiente específico de la capa.

        Args:
            dt (float): El paso de tiempo para la evolución.
        """
        with self.lock:
            for capa_idx in range(self.num_capas):
                alpha = self.alphas[capa_idx]
                phase_change = np.exp(-1j * alpha * dt)
                self.campo_q[capa_idx] *= phase_change


# --- Instancia Global y Lógica de Simulación ---
campo_toroidal_global = ToroidalField(NUM_CAPAS, NUM_FILAS, NUM_COLUMNAS)
stop_event = threading.Event()
try:
    # Usar las constantes globales para la instancia del servicio
    campo_toroidal_global_servicio = ToroidalField(
        num_capas=NUM_CAPAS,
        num_rows=NUM_FILAS,
        num_cols=NUM_COLUMNAS,
        alphas=None,  # Usará los defaults internos basados en num_capas
        dampings=None  # Usará los defaults internos basados en num_capas
    )
    logger.info(
        "Aplicando influencias iniciales al campo de "
        "confinamiento global (servicio)..."
    )
    campo_toroidal_global_servicio.aplicar_influencia(
        capa=0, row=1, col=2, vector=np.array([1.0, 0.5]),
        nombre_watcher="watcher_init_1"
    )
    campo_toroidal_global_servicio.aplicar_influencia(
        capa=2, row=3, col=0, vector=np.array([0.2, -0.1]),
        nombre_watcher="watcher_init_2"
    )
    logger.info("Influencias iniciales aplicadas al campo global (servicio).")

except ValueError:
    logger.exception(
        "Error crítico al inicializar ToroidalField global (servicio). "
        "Terminando."
    )
    exit(1)
except Exception as e:
    logger.exception(
        f"Error inesperado durante la inicialización del servicio: {e}"
    )
    exit(1)


# --- Hilo y Función de Simulación ---
simulation_thread = None
stop_simulation_event = threading.Event()


def simulation_loop(dt: float, beta: float):
    """Bucle que ejecuta la simulación dinámica periódicamente."""
    logger.info(f"Iniciando bucle de simulación ECU con dt={dt}, "
                f"beta={beta}...")
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
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Verifica la salud del servicio ECU (Experiencia de Campo Unificado).

    Incluye el estado de inicialización del campo toroidal y el hilo
    de simulación.
    """
    sim_alive = simulation_thread.is_alive() if simulation_thread else False
    service_ready = (
        campo_toroidal_global_servicio and
        hasattr(campo_toroidal_global_servicio, 'num_capas')
    )
    status_code = 503  # Service Unavailable por defecto
    response = {
        "status": "error",
        "message": "Servicio ECU no completamente inicializado.",
        "simulation_running": sim_alive,
        "field_initialized": service_ready
    }

    if service_ready and sim_alive:
        response["status"] = "success"
        response["message"] = "Servicio ECU saludable y simulación activa."
        status_code = 200
    elif service_ready and not sim_alive:
        response["status"] = "warning"
        response["message"] = ("Servicio ECU inicializado pero la "
                               "simulación no está activa.")
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
                "descripcion":
                    "Mapa de intensidad del campo toroidal ponderado por capa",
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
                "message": "Error interno del servidor al obtener estado unificado."
            }
        ), 500


@app.route("/api/ecu/influence", methods=["POST"])
def recibir_influencia_malla() -> Tuple[Any, int]:
    """
    Endpoint REST para recibir una influencia externa y aplicarla al campo.

    Espera un payload JSON con 'capa', 'row', 'col', 'vector' (lista 2D),
    y 'nombre_watcher'. Aplica el 'vector' como una perturbación al campo
    vectorial en la ubicación especificada.
    """
    logger.info("Solicitud POST /api/ecu/influence recibida.")
    try:
        data = request.get_json()
        if not data:
            logger.error(
                "No se recibió payload JSON en la solicitud "
                "POST /api/ecu/influence."
            )
            return jsonify({"status": "error",
                            "message": "Payload JSON vacío o ausente"}), 400
    except Exception as e_json:
        logger.error(
            "Error al parsear JSON en la solicitud "
            f"POST /api/ecu/influence: {e_json}"
        )
        return jsonify({"status": "error",
                        "message": "Error al parsear el payload JSON."}), 400

    required_fields: Dict[str, type] = {
        "capa": int, "row": int, "col": int,
        "vector": list, "nombre_watcher": str
    }
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        msg = (
            "Faltan campos requeridos en el JSON: "
            f"{', '.join(missing_fields)}"
        )
        logger.error(msg)
        return jsonify({"status": "error", "message": msg}), 400

    type_errors = []
    for field, expected_type in required_fields.items():
        # Asegurarse que el campo existe antes de verificar el tipo
        if field in data and not isinstance(data[field], expected_type):
            type_errors.append(
                f"Campo '{field}' debe ser {expected_type.__name__}, "
                f"recibido {type(data[field]).__name__}"
            )

    vector_data = data.get("vector")
    vector_complex = None
    if isinstance(vector_data, list) and len(vector_data) == 2:
        try:
            vector_complex = complex(float(vector_data[0]), float(vector_data[1]))
        except (ValueError, TypeError):
            type_errors.append("Campo 'vector' debe contener números.")
    elif 'vector' in data:
        type_errors.append("Campo 'vector' debe ser lista de 2 números")

    if type_errors:
        msg = "errores de tipo en json: " + "; ".join(type_errors).lower()
        logger.error(msg)
        return jsonify({"status": "error", "message": msg}), 400

    # Si llegamos aquí, los campos requeridos existen y tienen tipos
    # básicos correctos
    capa, row, col = data['capa'], data['row'], data['col']
    nombre_watcher = data['nombre_watcher']

    if not (0 <= capa < campo_toroidal_global_servicio.num_capas):
        return jsonify(
            {"status": "error",
            "message": "Índice de capa fuera de rango."
            }), 400
    if not (0 <= row < campo_toroidal_global_servicio.num_rows):
        return jsonify(
            {"status": "error",
            "message": "Índice de fila fuera de rango."
            }), 400
    if not (0 <= col < campo_toroidal_global_servicio.num_cols):
        return jsonify(
            {"status": "error",
            "message": "Índice de columna fuera de rango."
            }), 400

    try:
        success = campo_toroidal_global_servicio.aplicar_influencia(
            capa=capa, row=row, col=col,
            vector=vector_complex, nombre_watcher=nombre_watcher
        )
        if success:
            logger.info(
                f"Influencia de '{nombre_watcher}' aplicada exitosamente via"
                f"API en ({capa}, {row}, {col})."
            )
            return jsonify({
                "status": "success",
                "message": f"Influencia de '{nombre_watcher}' aplicada.",
                "applied_to": {"capa": capa, "row": row, "col": col},
                "vector": [vector_complex.real, vector_complex.imag]
            }), 200
        else:
            logger.error(
                f"Fallo al aplicar influencia de '{nombre_watcher}' via API "
                f"en ({capa}, {row}, {col}) por validación interna."
            )
            return jsonify({"status": "error",
                            "message": "Error de validación al aplicar "
                                       "influencia."}), 400
    except Exception as e_apply:
        logger.exception(
            "Error inesperado en endpoint /api/ecu/influence "
            f"al aplicar influencia: {e_apply}"
        )
        return jsonify({"status": "error",
                        "message": "Error interno al procesar la "
                                   "influencia."}), 500


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
            }
        }
        return jsonify(response_data), 200
    except Exception as e_field_vector:
        logger.exception(
            f"Error en endpoint /api/ecu/field_vector: {e_field_vector}"
        )
        error_response = {
            "status": "error",
            "message": "Error interno al procesar la solicitud del "
                       "campo vectorial."
        }
        return jsonify(error_response), 500


# --- Función Principal y Arranque ---
def main():
    global simulation_thread

    logger.info(
        f"Configuración ECU: {NUM_CAPAS}x{NUM_FILAS}x{NUM_COLUMNAS},"
        f"SimInterval={SIMULATION_INTERVAL}s, Beta={BETA_COUPLING}"
    )

    logger.info("Creando e iniciando hilo de simulación ECU...")
    stop_simulation_event.clear()
    simulation_thread = threading.Thread(
        target=simulation_loop,
        args=(SIMULATION_INTERVAL, BETA_COUPLING),
        daemon=True,
        name="ECUSimLoop"
    )
    simulation_thread.start()

    puerto_servicio = int(os.environ.get("MATRIZ_ECU_PORT", 8000))
    logger.info(
        f"Iniciando servicio Flask de ecu en puerto {puerto_servicio}..."
    )
    app.run(
        host="0.0.0.0", port=puerto_servicio,
        debug=False, use_reloader=False
    )


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
            simulation_thread.join(
                timeout=SIMULATION_INTERVAL * 2
            )
            if simulation_thread.is_alive():
                logger.warning(
                    "El hilo de simulación ECU no terminó limpiamente."
                )
        logger.info("Servicio matriz_ecu finalizado.")

# --- END OF FILE matriz_ecu.py (CORREGIDO) ---
