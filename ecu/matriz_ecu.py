#!/usr/bin/env python3
"""Módulo que define la simulación de un campo cimático toroidal.

Este módulo establece el marco conceptual y la implementación para simular un
campo cimático en una topología toroidal. La simulación modela cómo las ondas
se propagan, interfieren y disipan energía a través de un medio tridimensional
discretizado (grilla).

Conceptualmente, cada punto en la grilla representa un oscilador local con una
amplitud y una fase, descritas por un número complejo (ψ). La topología
toroidal implica que los bordes de la grilla se conectan, creando una
superficie continua sin fronteras. Este diseño es ideal para modelar campos
de ondas auto-contenidos y resonantes.

El propósito es estudiar la emergencia de patrones coherentes (análogos a las
figuras de Chladni en la cimática) a partir de la dinámica de ondas locales,
gobernada por una ecuación de onda con términos de acoplamiento y disipación.

Funcionalidad principal:
- Define la clase `ToroidalField` que encapsula el estado y la dinámica del campo.
- Ejecuta un bucle de simulación en un hilo de fondo para la evolución continua.
- Expone una API REST (a través de Flask) para interactuar con la simulación.
"""

import logging
import os
import sys
import threading
import time
from typing import Any, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, request

from .validator_ecu import InfluenceValidator


# --- Configuración del Logging ---
def setup_logging():
    """Configuración centralizada del logging."""
    logger = logging.getLogger("matriz_ecu")
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Nivel de logging configurable por entorno
    log_level = os.environ.get("ECU_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    return logger


logger = setup_logging()

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
    """Representa un campo de ondas cimáticas en una topología toroidal.

    Esta clase es la analogía principal de la simulación: representa un campo
    de ondas cimáticas evolucionando en un medio con propiedades específicas.
    La dinámica del campo está diseñada para simular la Ecuación de Onda en un
    medio con disipación, permitiendo la formación de patrones de interferencia
    complejos y auto-organizados.

    Atributos:
        campo_q (List[np.ndarray]): Una lista de arrays de NumPy, donde cada
            array representa una capa 2D del campo. Cada elemento del array es
            un número complejo (ψ) que codifica la amplitud y la fase de la
            onda en ese punto.
        propagation_coeffs (List[float]): Análogo a la velocidad de fase de la
            onda en cada capa del medio. Un valor más alto significa que la
            fase de la onda evoluciona más rápidamente.
        dissipation_coeffs (List[float]): Análogo a la atenuación de la onda
            en cada capa. Controla la tasa a la que la energía (amplitud) de
            la onda se disipa con el tiempo.
    """

    def __init__(
        self,
        num_capas: int,
        num_rows: int,
        num_cols: int,
        propagation_coeffs: Optional[List[float]] = None,
        dissipation_coeffs: Optional[List[float]] = None,
    ):
        """Inicializa el campo cimático, definiendo las propiedades del medio.

        Analogía: Este constructor define las propiedades fundamentales del
        "medio" en el que las ondas cimáticas se propagarán. Las dimensiones
        establecen la geometría del espacio, mientras que los coeficientes
        de propagación y disipación determinan cómo las ondas evolucionan y
        pierden energía en cada capa del medio.

        Args:
            num_capas (int): Número de capas en la grilla (profundidad).
            num_rows (int): Número de filas en la grilla (dimensión vertical).
            num_cols (int): Número de columnas en la grilla (dimensión azimutal).
            propagation_coeffs (Optional[List[float]]): Define la velocidad de
                fase de la onda por capa. Si es `None`, se usa un valor por
                defecto.
            dissipation_coeffs (Optional[List[float]]): Define la atenuación
                de la onda por capa. Si es `None`, se usa un valor por defecto.
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
            vector (complex): Influencia como número complejo (amplitud y fase).
            nombre_watcher (str): Nombre del watcher que aplica la influencia.

        Returns:
            bool: True si la influencia se aplicó, False en caso contrario.
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
                "Se necesita al menos 2 capas para calcular el gradiente entre capas."
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
            f"Gradiente entre capas calculado (shape: {gradiente_entre_capas.shape})"
        )
        return gradiente_entre_capas

    def get_energy_density_map(self) -> np.ndarray:
        """Genera un mapa de densidad de energía, análogo a los patrones de la cimática.

        Analogía: Este método calcula el análogo a los patrones visibles en los
        experimentos de cimática, como las Figuras de Chladni, que revelan las
        líneas nodales de un campo vibratorio.

        Ecuación Conceptual: La energía de una onda es proporcional al cuadrado
        de su amplitud (E ∝ |ψ|²). El mapa resultante agrega la energía en
        cada ubicación (fila, columna) a través de todas las capas, creando
        una representación 2D de la intensidad del campo.

        Returns:
            np.ndarray: Un array 2D (num_rows x num_cols) que representa el mapa
                de densidad de energía.
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
        """Avanza un paso en la dinámica de la onda.

        Este método implementa una solución numérica de la Ecuación de Onda 2D
        con disipación usando el método de diferencias finitas. Modela cómo las
        ondas se propagan, interfieren y disipan energía.

        Ecuación Física Implementada:
        La ecuación diferencial parcial (PDE) que este método resuelve es:
        ```latex
        \frac{\partial^2 \psi}{\partial t^2} = c^2 \nabla^2 \psi - \gamma \frac{\partial \psi}{\partial t}
        ```

        Donde:
        - ψ: Es el campo de ondas (amplitud y fase).
        - c: Es la velocidad de propagación, relacionada con `propagation_coeffs`.
        - ∇²: Es el operador Laplaciano, que representa la curvatura del campo.
        - γ: Es el coeficiente de disipación, relacionado con `dissipation_coeffs`.

        Correspondencia de la Implementación:
        La solución numérica se descompone en los siguientes términos:
        - `damped`: Corresponde al término de disipación (-γ(∂ψ/∂t)), que reduce
          la amplitud de la onda con el tiempo.
        - `advected`: Representa una parte del término de propagación (c²∇²ψ),
          simulando el movimiento de la onda en una dirección.
        - `coupled`: Representa otra parte del término de propagación (c²∇²ψ),
          modelando la interacción (acoplamiento) con los nodos vecinos.

        Args:
            dt (float): Paso de tiempo para la integración numérica.
            beta (float): Factor de acoplamiento que controla la influencia
                entre capas adyacentes.
        """
        if beta < 0:
            logger.warning(
                "El factor beta (acoplamiento vertical) debería ser no negativo."
            )

        with self.lock:
            # Precalcular arrays para broadcasting
            propagation_coeffs_array = np.array(self.propagation_coeffs)[
                :, np.newaxis, np.newaxis
            ]
            dissipation_coeffs_array = np.array(self.dissipation_coeffs)[
                :, np.newaxis, np.newaxis
            ]

            # Crear array 3D para operaciones vectorizadas
            campo_3d = np.stack(self.campo_q)

            # Calcular vecinos con roll (condiciones toroidales)
            v_left = np.roll(campo_3d, shift=1, axis=2)
            v_up = np.roll(campo_3d, shift=1, axis=1)
            v_down = np.roll(campo_3d, shift=-1, axis=1)

            # Cálculos vectorizados
            damped = campo_3d * (1.0 - dissipation_coeffs_array * dt)
            advected = propagation_coeffs_array * v_left * dt
            coupled = beta * (v_up + v_down) * dt

            # Actualizar campo
            nuevo_campo_3d = damped + advected + coupled

            # Convertir de vuelta a lista de arrays 2D
            self.campo_q = [nuevo_campo_3d[i] for i in range(self.num_capas)]

    def set_uniform_potential_field(self, seed: Optional[int] = None):
        """Inicializa el campo a un estado de potencial uniforme pero incoherente.

        Analogía: Este proceso es análogo a una superficie de agua en reposo
        que, aunque parece macroscópicamente plana (potencial uniforme),
        posee fluctuaciones de fase microscópicas en cada punto. Cada nodo se
        inicializa con una amplitud de 1 y una fase aleatoria, representando
        un estado de máxima energía potencial pero sin coherencia global.

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
        """Simula la evolución intrínseca de la fase de la onda en cada punto.

        Analogía: Este método simula la evolución temporal intrínseca de la
        fase de la onda en cada punto del medio, gobernada por las propiedades
        locales (el coeficiente de propagación de la capa). Es como si cada
        punto del campo fuera un reloj que avanza a su propio ritmo.

        Ecuación Conceptual: Aplica una rotación en el plano complejo a cada
        punto del campo, lo que corresponde a la solución de la Ecuación de
        Schrödinger para una partícula libre en su forma discreta:
        ψ(t+dt) = e^(-i * α * dt) * ψ(t)
        donde α es el `propagation_coeff` de la capa.

        Args:
            dt (float): El paso de tiempo para la evolución de la fase.
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


def cymatic_simulation_loop_adaptive(dt: float, beta: float):
    """Bucle de simulación con paso de tiempo adaptable."""
    logger.info(f"Iniciando bucle de simulación adaptativa con dt={dt}, beta={beta}...")

    # Estadísticas de rendimiento
    simulation_times = []

    while not stop_simulation_event.is_set():
        try:
            start_time = time.monotonic()

            # Ejecutar paso de simulación
            campo_toroidal_global_servicio.apply_wave_dynamics_step(dt, beta)
            campo_toroidal_global_servicio.apply_internal_phase_evolution(dt)

            # Calcular tiempo de ejecución
            elapsed = time.monotonic() - start_time
            simulation_times.append(elapsed)

            # Mantener solo las últimas 10 mediciones
            if len(simulation_times) > 10:
                simulation_times.pop(0)

            # Ajustar dt si es necesario (no menos del 50% del valor original)
            avg_time = (
                sum(simulation_times) / len(simulation_times)
                if simulation_times
                else elapsed
            )
            if avg_time > dt * 0.8:  # Si usa más del 80% del tiempo disponible
                new_dt = dt * 0.9  # Reducir dt un 10%
                logger.info(f"Ajustando dt de {dt} a {new_dt} por sobrecarga")
                dt = max(
                    new_dt, SIMULATION_INTERVAL * 0.5
                )  # No menos de la mitad del original

            sleep_time = max(0, dt - elapsed)
            stop_simulation_event.wait(sleep_time)

        except Exception as e:
            logger.error("Error en el bucle de simulación: %s", e, exc_info=True)
            stop_simulation_event.wait(5)  # Esperar antes de reintentar


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


@app.route("/api/ecu/energy", methods=["GET"])
def obtener_mapa_energia() -> Tuple[Any, int]:
    """
    Obtiene el mapa de densidad de energía del campo cimático.

    Returns:
        JSON con el mapa de energía y metadatos
    """
    try:
        energy_map = campo_toroidal_global_servicio.get_energy_density_map()
        response_data = {
            "status": "success",
            "data": {"energy_density_map": energy_map.tolist(), "type": "energy_map"},
            "metadata": {
                "description": "Mapa de densidad de energía del campo cimático",
                "layers": campo_toroidal_global_servicio.num_capas,
                "rows": campo_toroidal_global_servicio.num_rows,
                "columns": campo_toroidal_global_servicio.num_cols,
                "timestamp": time.time(),
            },
        }
        logger.info("Mapa de densidad de energía calculado y enviado.")
        return jsonify(response_data), 200
    except Exception as e:
        logger.exception("Error en endpoint /api/ecu/energy: %s", e)
        return jsonify(
            {
                "status": "error",
                "message": "Error interno del servidor al obtener el mapa de energía.",
                "error_code": "INTERNAL_SERVER_ERROR",
            }
        ), 500


@app.route("/api/ecu/influence", methods=["POST"])
def recibir_influencia_malla() -> Tuple[Any, int]:
    """Recibe y aplica una influencia externa (perturbación) al campo cimático."""
    logger.info("Solicitud POST /api/ecu/influence recibida.")
    data = request.get_json()
    if not data:
        return jsonify(
            {"status": "error", "message": "Payload JSON vacío o ausente"}
        ), 400

    required_fields = ["capa", "row", "col", "vector", "nombre_watcher"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify(
            {"status": "error", "message": f"Faltan campos: {', '.join(missing)}"}
        ), 400

    try:
        capa, row, col = data["capa"], data["row"], data["col"]
        coord_errors = InfluenceValidator.validate_coordinates(
            capa, row, col, campo_toroidal_global_servicio
        )
        if coord_errors:
            return jsonify({"status": "error", "message": "; ".join(coord_errors)}), 400

        vector, vec_error = InfluenceValidator.validate_vector(data["vector"])
        if vec_error:
            return jsonify({"status": "error", "message": vec_error}), 400

        success = campo_toroidal_global_servicio.aplicar_influencia(
            capa=capa,
            row=row,
            col=col,
            vector=vector,
            nombre_watcher=data["nombre_watcher"],
        )

        if success:
            return jsonify(
                {
                    "status": "success",
                    "message": f"Influencia de '{data['nombre_watcher']}' aplicada.",
                    "applied_to": {"capa": capa, "row": row, "col": col},
                    "vector": [vector.real, vector.imag],
                }
            ), 200
        else:
            # Este caso puede ser redundante si la validación es exhaustiva
            return jsonify(
                {"status": "error", "message": "Error interno al aplicar influencia."}
            ), 400

    except (TypeError, ValueError) as e:
        logger.warning("Error de tipo/valor en payload de influencia: %s", e)
        return jsonify(
            {
                "status": "error",
                "message": "Error en el formato de los datos del payload.",
            }
        ), 400
    except Exception as e:
        logger.exception("Error inesperado en endpoint /api/ecu/influence: %s", e)
        return jsonify(
            {"status": "error", "message": "Error interno del servidor."}
        ), 500


@app.route("/api/ecu/field_vector", methods=["GET"])
def get_field_vector_paginated() -> Tuple[Any, int]:
    """
    Endpoint con soporte para paginación de campos grandes.

    Query Parameters:
        page: Número de página (por defecto 1)
        per_page: Elementos por página (por defecto 100)
        layer: Filtro opcional por capa específica
    """
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 100, type=int)
        layer_filter = request.args.get("layer", type=int)

        with campo_toroidal_global_servicio.lock:
            if layer_filter is not None:
                if not 0 <= layer_filter < campo_toroidal_global_servicio.num_capas:
                    return jsonify(
                        {
                            "status": "error",
                            "message": f"La capa {layer_filter} está fuera de rango",
                        }
                    ), 400

                # Paginación sobre filas de una capa específica
                layer_data = campo_toroidal_global_servicio.campo_q[layer_filter]
                total_items = campo_toroidal_global_servicio.num_rows
                start_index = (page - 1) * per_page
                end_index = start_index + per_page
                paginated_rows = layer_data[start_index:end_index]

                campo_data = [
                    [[cell.real, cell.imag] for cell in row] for row in paginated_rows
                ]

            else:
                # Paginación sobre las capas completas
                total_items = campo_toroidal_global_servicio.num_capas
                start_index = (page - 1) * per_page
                end_index = start_index + per_page
                paginated_layers = campo_toroidal_global_servicio.campo_q[
                    start_index:end_index
                ]

                campo_data = [
                    [[[cell.real, cell.imag] for cell in row] for row in layer]
                    for layer in paginated_layers
                ]

        total_pages = (total_items + per_page - 1) // per_page
        if page > total_pages and total_items > 0:
            return jsonify({"status": "error", "message": "Página fuera de rango"}), 404

        return jsonify(
            {
                "status": "success",
                "data": campo_data,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total_items": total_items,
                    "total_pages": total_pages,
                },
            }
        ), 200

    except Exception as e:
        logger.exception("Error en endpoint paginado: %s", e)
        return jsonify(
            {"status": "error", "message": "Error interno al procesar la solicitud"}
        ), 500


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
    """Función principal que arranca la aplicación y el bucle de simulación."""
    global simulation_thread

    logger.info(
        f"Configuración de simulación: {NUM_CAPAS}x{NUM_FILAS}x{NUM_COLUMNAS}, "
        f"Intervalo={SIMULATION_INTERVAL}s, Beta={BETA_COUPLING}"
    )

    logger.info("Creando e iniciando hilo de simulación adaptativa...")
    stop_simulation_event.clear()
    simulation_thread = threading.Thread(
        target=cymatic_simulation_loop_adaptive,
        args=(SIMULATION_INTERVAL, BETA_COUPLING),
        daemon=True,
        name="CymaticSimLoop",
    )
    simulation_thread.start()

    puerto_servicio = int(os.environ.get("MATRIZ_ECU_PORT", 8000))
    logger.info(f"Iniciando servicio Flask en puerto {puerto_servicio}...")
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
                logger.warning("El hilo de simulación cimática no terminó limpiamente.")
        logger.info("Servicio de simulación cimática finalizado.")
