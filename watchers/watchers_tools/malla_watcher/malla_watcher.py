#!/usr/bin/env python3
"""Define una malla hexagonal cilíndrica y su interacción con un campo externo.

Este módulo modela una malla hexagonal cilíndrica, inspirada en la estructura
del grafeno, como un sistema de osciladores acoplados. La malla interactúa
con un campo vectorial externo (proveniente de la experiencia de campo
unificada, ECU) que modula el acoplamiento entre los osciladores.
Además, la malla genera influencias sobre dicho campo, basadas en
la tasa de cambio del flujo del campo a través de ella, una analogía
con la inducción electromagnética.

Componentes Principales:
    Cell (importada): Representa un oscilador individual en la malla,
        caracterizado por su estado (amplitud, velocidad) y el campo externo
        local (q_vector) que experimenta.
    HexCylindricalMesh (importada): Gestiona la estructura de la malla
        hexagonal cilíndrica, incluyendo la validación de la conectividad
        entre celdas y la implementación de condiciones de contorno periódicas.
    PhosWave: Modela el mecanismo de acoplamiento entre celdas (osciladores).
    Electron: Modela el mecanismo de amortiguación local en cada celda.

Interacciones Fundamentales:
    1. ECU → Malla: El sistema obtiene periódicamente el campo vectorial
       desde la ECU y lo aplica a las celdas de la malla mediante un proceso
       de interpolación.
    2. Malla → ECU: El sistema envía influencias a la ECU (específicamente
       al componente toroide), basadas en la tasa de cambio del flujo del
       campo de la ECU a través de la malla (dΦ/dt).

Dependencias Clave:
    `cilindro_grafenal.HexCylindricalMesh`: Clase central para la generación
    y validación de la estructura digital del cilindro de grafeno simulado.
    `numpy`: Para operaciones numéricas eficientes, especialmente en el manejo
    de vectores y campos.
    `requests`: Para la comunicación HTTP con otros servicios (ECU, AgentAI).
    `Flask`: Expone una API que permite el control y monitoreo de la malla.
"""

import json
import logging
import math
import os
import threading
import time
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from flask import Flask, jsonify, request
from scipy.interpolate import RegularGridInterpolator
from werkzeug.exceptions import HTTPException

from watchers.watchers_tools.malla_watcher.utils.cilindro_grafenal import (
    Cell,
    HexCylindricalMesh,
)

# --- Configuración del Logging ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("malla_watcher")
if not logger.hasHandlers():
    handler = logging.FileHandler(os.path.join(log_dir, "malla_watcher.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    # Los logs DEBUG en métodos como _initialize_mesh o get_neighbor_cells
    # solo se mostrarán si el nivel del logger se cambia a DEBUG (ej.
    # desarrollo)

# Pequeña tolerancia para comparaciones de flotantes
EPSILON = 1e-9

# --- Constantes de Configuración para Comunicación ---
MATRIZ_ECU_BASE_URL = os.environ.get("MATRIZ_ECU_URL", "http://ecu:8000")
TORUS_NUM_CAPAS = int(os.environ.get("TORUS_NUM_CAPAS", 3))
TORUS_NUM_FILAS = int(os.environ.get("TORUS_NUM_FILAS", 4))
TORUS_NUM_COLUMNAS = int(os.environ.get("TORUS_NUM_COLUMNAS", 5))
# Umbral para métrica de actividad
AMPLITUDE_INFLUENCE_THRESHOLD = float(os.environ.get("MW_INFLUENCE_THRESHOLD", 5.0))
# Valor para normalizar métrica de actividad
MAX_AMPLITUDE_FOR_NORMALIZATION = float(os.environ.get("MW_MAX_AMPLITUDE_NORM", 20.0))
REQUESTS_TIMEOUT = float(os.environ.get("MW_REQUESTS_TIMEOUT", 2.0))

# --- Constantes de Configuración para Control ---
BASE_COUPLING_T = float(os.environ.get("MW_BASE_T", 0.6))
K_GAIN_COUPLING = float(os.environ.get("MW_K_GAIN_T", 0.1))
BASE_DAMPING_E = float(os.environ.get("MW_BASE_E", 0.1))
K_GAIN_DAMPING = float(os.environ.get("MW_K_GAIN_E", 0.05))

# --- Constantes de Configuración para Simulación ---
SIMULATION_INTERVAL = float(
    os.environ.get("MW_SIM_INTERVAL", 0.5)
)  # Segundos (esto es dt)
# Umbral de |dPhi/dt| para enviar influencia
DPHI_DT_INFLUENCE_THRESHOLD = float(os.environ.get("MW_DPHI_DT_THRESHOLD", 1.0))

# --- Constantes para Claves de Diccionario y Payloads ---
# Claves para el estado agregado
KEY_AVG_AMPLITUDE = "avg_amplitude"
KEY_MAX_AMPLITUDE = "max_amplitude"
KEY_AVG_VELOCITY = "avg_velocity"
KEY_MAX_VELOCITY = "max_velocity"
KEY_AVG_KINETIC_ENERGY = "avg_kinetic_energy"
KEY_MAX_KINETIC_ENERGY = "max_kinetic_energy"
KEY_AVG_ACTIVITY_MAGNITUDE = "avg_activity_magnitude"
KEY_MAX_ACTIVITY_MAGNITUDE = "max_activity_magnitude"
KEY_CELLS_OVER_THRESHOLD = "cells_over_threshold"
# Claves para parámetros de control
KEY_PHOSWAVE_C = "phoswave_C"
KEY_ELECTRON_D = "electron_D"
# Claves comunes de API
KEY_STATUS = "status"
KEY_MESSAGE = "message"
KEY_DETAILS = "details"
KEY_ERROR = "error"
KEY_SUCCESS = "success"
KEY_WARNING = "warning"
# Claves para payloads de influencia
KEY_CAPA = "capa"
KEY_ROW = "row"
KEY_COL = "col"
KEY_VECTOR = "vector"
KEY_NOMBRE_WATCHER = "nombre_watcher"
# Claves para registro en AgentAI
KEY_NOMBRE = "nombre"
KEY_URL = "url"
KEY_URL_SALUD = "url_salud"
KEY_TIPO = "tipo"
KEY_APORTA_A = "aporta_a"
KEY_NATURALEZA_AUXILIAR = "naturaleza_auxiliar"
KEY_DESCRIPCION = "descripcion"


# --- Caché Global ---
_CACHED_GEOMETRY = {}

# --- Clases PhosWave y Electron ---
@dataclass
class PhosWave:
    """Representa el mecanismo de acoplamiento entre celdas (osciladores).

    Esta clase encapsula el coeficiente de acoplamiento (C), análogo a la
    energía de interacción que media la propagación de fonones (vibraciones
    cuantizadas de la red) en un cristal. Un valor de C más alto implica una
    interacción más fuerte entre celdas vecinas.

    Args:
        coef_acoplamiento (float): Coeficiente de acoplamiento inicial.

    Attributes:
        C (float): El coeficiente de acoplamiento. Debe ser no negativo.
    """

    coef_acoplamiento: InitVar[float] = BASE_COUPLING_T
    C: float = field(init=False)

    def __post_init__(self, coef_acoplamiento: float):
        """Asegura que el coeficiente de acoplamiento no sea negativo."""
        self.C = max(0.0, coef_acoplamiento)

    def ajustar_coeficientes(self, nuevos_C: float):
        """Ajusta el coeficiente de acoplamiento.

        Args:
            nuevos_C (float): El nuevo valor para el coeficiente de
                acoplamiento. Se asegura que el valor almacenado no sea
                negativo.
        """
        self.C = max(0.0, nuevos_C)
        logger.debug("PhosWave coeficiente de acoplamiento ajustado a C=%.3f", self.C)


@dataclass
class Electron:
    """
    Representa el mecanismo de amortiguación local en cada celda (oscilador).

    Esta clase encapsula el coeficiente de amortiguación (D). Físicamente,
    es análogo a un término de arrastre o fricción que disipa energía del
    oscilador, similar a la dispersión de electrones (scattering) en una red
    cristalina que resulta en una pérdida de energía de las vibraciones.

    Args:
        coef_amortiguacion (float): Coeficiente de amortiguación inicial.

    Attributes:
        D (float): El coeficiente de amortiguación. Debe ser no negativo.
    """

    coef_amortiguacion: InitVar[float] = BASE_DAMPING_E
    D: float = field(init=False)

    def __post_init__(self, coef_amortiguacion: float):
        """Asegura que el coeficiente de amortiguación no sea negativo."""
        self.D = max(0.0, coef_amortiguacion)

    def ajustar_coeficientes(self, nuevos_D: float):
        """Ajusta el coeficiente de amortiguación.

        Args:
            nuevos_D (float): El nuevo valor para el coeficiente de
                amortiguación. Se asegura que el valor almacenado no sea
                negativo.
        """
        self.D = max(0.0, nuevos_D)
        logger.debug("Electron coeficiente de amortiguación ajustado a D=%.3f", self.D)


def apply_external_field_to_mesh(
    mesh_instance: HexCylindricalMesh, field_vector_map: List[List[List[List[float]]]]
) -> None:
    """Aplica un campo vectorial externo a las celdas de la malla de forma
    vectorizada.

    Este campo, típicamente proveniente de la ECU, se interpola para
    actualizar el atributo `q_vector` de cada celda en la instancia de
    malla proporcionada. El `q_vector` representa el campo local que
    experimenta la celda.

    Args:
        mesh_instance (HexCylindricalMesh): La instancia de la malla cuyas
            celdas serán actualizadas.
        field_vector_map (List[List[List[List[float]]]]): Una estructura de
            datos anidada (se espera que sea de 4 dimensiones:
            [capas, filas, columnas, 2]) que representa el campo vectorial
            externo.
    """
    if not mesh_instance or not mesh_instance.cells:
        logger.warning(
            "Intento de aplicar campo externo a malla no inicializada o vacía."
        )
        return

    all_mesh_cells = mesh_instance.get_all_cells()
    if not all_mesh_cells:
        logger.warning("No hay celdas para aplicar campo externo.")
        return

    try:
        field_vector_np = np.array(field_vector_map, dtype=float)
        if field_vector_np.ndim != 4 or field_vector_np.shape[-1] != 2:
            logger.error(
                "El campo vectorial externo no tiene el shape esperado "
                "[capas, filas, columnas, 2]. Recibido: %s",
                field_vector_np.shape,
            )
            return
    except (ValueError, TypeError) as err:
        logger.error("Error al convertir field_vector_map a NumPy array: %s", err)
        return

    num_capas_torus, num_rows_torus, num_cols_torus, _ = field_vector_np.shape
    if num_rows_torus <= 1 or num_cols_torus <= 1:
        logger.error(
            "La interpolación requiere dimensiones de al menos 2x2 en el "
            "campo. Recibido: %s",
            field_vector_np.shape,
        )
        return

    # Usar solo la primera capa del campo, como en la lógica original
    field_slice = field_vector_np[0, :, :, :]

    # 1. Extraer coordenadas de las celdas de forma vectorizada
    thetas = np.array([cell.theta for cell in all_mesh_cells])
    zs = np.array([cell.z for cell in all_mesh_cells])

    # 2. Mapear coordenadas de la malla a índices de la red del toroide
    col_coords = (thetas / (2 * math.pi)) * (num_cols_torus - 1)

    row_coords = np.zeros_like(zs)
    cylinder_height = mesh_instance.max_z - mesh_instance.min_z
    if cylinder_height > EPSILON and num_rows_torus > 1:
        normalized_z = (zs - mesh_instance.min_z) / cylinder_height
        row_coords = normalized_z * (num_rows_torus - 1)

    # Asegurar que las coordenadas estén dentro de los límites de la red
    row_coords = np.clip(row_coords, 0, num_rows_torus - 1)
    col_coords = np.clip(col_coords, 0, num_cols_torus - 1)

    # 3. Crear el interpolador
    # Los puntos de la red son los índices de las filas y columnas
    grid_points = (np.arange(num_rows_torus), np.arange(num_cols_torus))
    # El método de interpolación 'linear' es equivalente a bilineal para 2D
    interpolator = RegularGridInterpolator(
        grid_points, field_slice, method="linear", bounds_error=False, fill_value=None
    )

    # 4. Interpolar todos los puntos a la vez
    # El interpolador espera un array de shape (n_points, n_dims)
    points_to_interpolate = np.vstack((row_coords, col_coords)).T
    interpolated_q_vectors = interpolator(points_to_interpolate)

    # 5. Asignar los nuevos q_vectors a las celdas
    for i, cell in enumerate(all_mesh_cells):
        cell.q_vector = interpolated_q_vectors[i]

    logger.debug(
        "Campo vectorial externo aplicado a %d celdas mediante "
        "interpolación vectorizada.",
        len(all_mesh_cells),
    )


# --- Instancia Global de la Malla Cilíndrica ---
MESH_RADIUS = float(os.environ.get("MW_RADIUS", 5.0))
MESH_HEIGHT_SEGMENTS = int(os.environ.get("MW_HEIGHT_SEG", 2))
MESH_CIRCUMFERENCE_SEGMENTS = int(os.environ.get("MW_CIRCUM_SEG", 6))
MESH_HEX_SIZE = float(os.environ.get("MW_HEX_SIZE", 1.0))
MESH_PERIODIC_Z = os.environ.get("MW_PERIODIC_Z", "True").lower() == "true"

# La inicialización global usará la nueva definición de Cell
try:
    malla_cilindrica_global = HexCylindricalMesh(
        radius=float(os.environ.get("MW_RADIUS", 5.0)),
        height_segments=int(os.environ.get("MW_HEIGHT_SEG", 3)),
        circumference_segments_target=int(os.environ.get("MW_CIRCUM_SEG", 6)),
        hex_size=float(os.environ.get("MW_HEX_SIZE", 1.0)),
        periodic_z=os.environ.get("MW_PERIODIC_Z", "True").lower() == "true",
    )
    # Inicializar el flujo previo
    malla_cilindrica_global.previous_flux = 0.0
    logger.info(
        "Instancia global inicial (import time) creada con %d celdas.",
        len(malla_cilindrica_global.cells),
    )
except Exception as err:
    logger.exception(
        "Error al inicializar HexCylindricalMesh global (import time). "
        "Será re-inicializada en __main__.",
        exc_info=err,
    )
    malla_cilindrica_global = None


# --- Instancias de PhosWave y Electron ---
# Inicializadas con valores base.
resonador_global = PhosWave(coef_acoplamiento=float(os.environ.get("MW_BASE_T", 0.6)))
electron_global = Electron(coef_amortiguacion=float(os.environ.get("MW_BASE_E", 0.1)))


# --- Estado Agregado y Control ---
aggregate_state_lock = threading.Lock()
aggregate_state: Dict[str, Any] = {
    KEY_AVG_AMPLITUDE: 0.0,
    KEY_MAX_AMPLITUDE: 0.0,
    KEY_AVG_VELOCITY: 0.0,
    KEY_MAX_VELOCITY: 0.0,
    KEY_AVG_KINETIC_ENERGY: 0.0,
    KEY_MAX_KINETIC_ENERGY: 0.0,
    KEY_AVG_ACTIVITY_MAGNITUDE: 0.0,
    KEY_MAX_ACTIVITY_MAGNITUDE: 0.0,
    KEY_CELLS_OVER_THRESHOLD: 0,
}
control_lock = threading.Lock()
control_params: Dict[str, float] = {
    KEY_PHOSWAVE_C: resonador_global.C,
    KEY_ELECTRON_D: electron_global.D,
}


# --- Lógica de Simulación (Propagación y Estabilización) ---
def simular_paso_malla() -> None:
    """Simula un paso de la dinámica de osciladores acoplados en la malla.

    Este método modela la malla como un sistema de osciladores armónicos
    acoplados. La ecuación de movimiento para cada oscilador 'i' se rige por:
    ```latex
    m \frac{d^2x_i}{dt^2} = \sum_{j \in vecinos} k_{\text{mod}}(q_i) (x_j - x_i) - b \frac{dx_i}{dt}
    ```
    donde 'x' es la amplitud, 'k_mod' es el acoplamiento modulado por el
    campo externo, y 'b' es la amortiguación.

    Se utiliza el método de integración de Euler, un esquema numérico de
    primer orden, para resolver esta ecuación diferencial en pasos discretos 'dt'.
    La fuerza neta (acoplamiento + amortiguación) determina la aceleración,
    que a su vez actualiza la velocidad y luego la amplitud de cada celda.

    No recibe argumentos ni retorna valores, pero modifica el estado (amplitud,
    velocidad) de las celdas en `malla_cilindrica_global`.
    """
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        logger.error("Intento de simular paso en malla no inicializada o vacía.")
        return

    dt = SIMULATION_INTERVAL
    all_cells = mesh.get_all_cells()
    num_cells = len(all_cells)
    if num_cells == 0:
        return

    next_amplitudes = [0.0] * num_cells
    next_velocities = [0.0] * num_cells

    # Asumimos masa = 1 para simplificar: Force = Acceleration
    for i, cell_i in enumerate(all_cells):
        # 1a: Fuerza de Acoplamiento (de vecinos)
        coupling_force = 0.0
        neighbors = mesh.get_neighbor_cells(cell_i.q_axial, cell_i.r_axial)

        # Coeficiente de acoplamiento modulado por la MAGNITUD del campo local
        modulator = 1.0 + np.linalg.norm(cell_i.q_vector)
        modulated_coupling_C = resonador_global.C * max(0.0, modulator)

        for cell_j in neighbors:
            coupling_force += modulated_coupling_C * (
                cell_j.amplitude - cell_i.amplitude
            )

        # 1b: Fuerza de Amortiguación (local)
        damping_force = -electron_global.D * cell_i.velocity

        # 1c: Fuerza Neta
        net_force = coupling_force + damping_force

        # 1d: Calcular nueva velocidad (Euler)
        acceleration = net_force  # masa = 1
        next_velocities[i] = cell_i.velocity + acceleration * dt

    # Fase 2: Calcular nuevas amplitudes usando las nuevas velocidades
    for i, cell_i in enumerate(all_cells):
        next_amplitudes[i] = cell_i.amplitude + next_velocities[i] * dt

    # Fase 3: Actualizar los estados de las celdas
    for i, cell_i in enumerate(all_cells):
        cell_i.amplitude = next_amplitudes[i]
        cell_i.velocity = next_velocities[i]

    logger.debug("Paso de simulación completado para %d celdas.", num_cells)


def update_aggregate_state() -> None:
    """Calcula y actualiza el estado agregado de la malla.

    Calcula métricas promedio y máximas sobre todas las celdas, como amplitud,
    velocidad y energía cinética. La energía cinética (KE) de cada celda se
    calcula como KE = 0.5 * m * v², asumiendo una masa normalizada m=1.
    También calcula una 'magnitud de actividad' y cuenta cuántas celdas
    superan un umbral de actividad predefinido.

    Los resultados se almacenan en el diccionario global `aggregate_state`
    de forma segura para hilos.

    No recibe argumentos ni retorna valores, pero modifica el diccionario
    `aggregate_state`.
    """
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        with aggregate_state_lock:
            aggregate_state[KEY_AVG_AMPLITUDE] = 0.0
            aggregate_state[KEY_MAX_AMPLITUDE] = 0.0
            aggregate_state[KEY_AVG_VELOCITY] = 0.0
            aggregate_state[KEY_MAX_VELOCITY] = 0.0
            aggregate_state[KEY_AVG_KINETIC_ENERGY] = 0.0
            aggregate_state[KEY_MAX_KINETIC_ENERGY] = 0.0
            aggregate_state[KEY_AVG_ACTIVITY_MAGNITUDE] = 0.0
            aggregate_state[KEY_MAX_ACTIVITY_MAGNITUDE] = 0.0
            aggregate_state[KEY_CELLS_OVER_THRESHOLD] = 0
        logger.debug("Malla no inicializada o vacía, estado agregado reseteado.")
        return

    all_cells = mesh.get_all_cells()
    if not all_cells:
        with aggregate_state_lock:
            aggregate_state[KEY_AVG_AMPLITUDE] = 0.0
            aggregate_state[KEY_MAX_AMPLITUDE] = 0.0
            aggregate_state[KEY_AVG_VELOCITY] = 0.0
            aggregate_state[KEY_MAX_VELOCITY] = 0.0
            aggregate_state[KEY_AVG_KINETIC_ENERGY] = 0.0
            aggregate_state[KEY_MAX_KINETIC_ENERGY] = 0.0
            aggregate_state[KEY_AVG_ACTIVITY_MAGNITUDE] = 0.0
            aggregate_state[KEY_MAX_ACTIVITY_MAGNITUDE] = 0.0
            aggregate_state[KEY_CELLS_OVER_THRESHOLD] = 0
        logger.debug("Lista de celdas vacía, estado agregado reseteado.")
        return

    # Calcular métricas individuales
    amplitudes = [cell.amplitude for cell in all_cells]
    velocities = [cell.velocity for cell in all_cells]
    kinetic_energies = [0.5 * v**2 for v in velocities]
    activity_magnitudes = [
        math.sqrt(cell.amplitude**2 + cell.velocity**2) for cell in all_cells
    ]

    num_cells = len(all_cells)
    avg_amp = sum(amplitudes) / num_cells
    max_amp = max(amplitudes, default=0.0)
    avg_vel = sum(velocities) / num_cells
    max_vel = max(velocities, default=0.0)
    avg_ke = sum(kinetic_energies) / num_cells
    max_ke = max(kinetic_energies, default=0.0)
    avg_activity = sum(activity_magnitudes) / num_cells
    max_activity = max(activity_magnitudes, default=0.0)

    # El umbral de influencia ahora se compara con la métrica de actividad
    # total
    over_thresh = sum(
        1 for mag in activity_magnitudes if mag > AMPLITUDE_INFLUENCE_THRESHOLD
    )

    with aggregate_state_lock:
        aggregate_state[KEY_AVG_AMPLITUDE] = avg_amp
        aggregate_state[KEY_MAX_AMPLITUDE] = max_amp
        aggregate_state[KEY_AVG_VELOCITY] = avg_vel
        aggregate_state[KEY_MAX_VELOCITY] = max_vel
        aggregate_state[KEY_AVG_KINETIC_ENERGY] = avg_ke
        aggregate_state[KEY_MAX_KINETIC_ENERGY] = max_ke
        aggregate_state[KEY_AVG_ACTIVITY_MAGNITUDE] = avg_activity
        aggregate_state[KEY_MAX_ACTIVITY_MAGNITUDE] = max_activity
        aggregate_state[KEY_CELLS_OVER_THRESHOLD] = over_thresh

    logger.debug(
        "Estado agregado actualizado: AvgAmp=%.3f, MaxAmp=%.3f, AvgVel=%.3f, "
        "MaxVel=%.3f, AvgKE=%.3f, MaxKE=%.3f, AvgActivity=%.3f, "
        "MaxActivity=%.3f, OverThresh=%d",
        avg_amp,
        max_amp,
        avg_vel,
        max_vel,
        avg_ke,
        max_ke,
        avg_activity,
        max_activity,
        over_thresh,
    )


def calculate_flux(mesh: HexCylindricalMesh) -> float:
    """Calcula el flujo del campo vectorial 'q_vector' a través de la malla.

    En física, el flujo de un campo vectorial a través de una superficie es una
    medida de cuánto campo atraviesa dicha superficie. Se calcula mediante la
    integral de superficie del producto punto entre el campo vectorial y el
    vector normal a la superficie.

    ```latex
    \Phi = \int_S \vec{q} \cdot d\vec{A}
    ```

    La influencia que la malla ejerce de vuelta sobre `matriz_ecu` es
    proporcional a la tasa de cambio de este flujo (`dΦ/dt`).

    Donde:
    - Φ es el flujo.
    - F es el campo vectorial (en nuestro caso, `q_vector`).
    - dA es el vector de área diferencial, normal a la superficie.

    Para esta malla cilíndrica, aproximamos el cálculo de la siguiente manera:
    1. Cada celda hexagonal se trata como una pequeña superficie plana.
    2. El vector de área (dA) de cada celda tiene una magnitud igual al área
       del hexágono y una dirección normal a la superficie del cilindro en la
       posición de la celda.
    3. El flujo total es la suma de los productos punto del `q_vector` de cada
       celda con su vector de área.

    Args:
        mesh (HexCylindricalMesh): La instancia de la malla.

    Returns:
        float: El valor total del flujo calculado.
    """
    global _CACHED_GEOMETRY

    if not mesh or not mesh.cells:
        return 0.0

    # Usamos propiedades estructurales para la clave de caché para evitar colisiones de id()
    # y detectar cambios reales en la geometría
    cache_key = (mesh.hex_size, len(mesh.cells), mesh.radius)

    if cache_key not in _CACHED_GEOMETRY:
        # Solo usamos cells para calcular la geometría estática
        cells_for_geom = mesh.get_all_cells()
        if not cells_for_geom:
             return 0.0

        # Área de un hexágono regular de lado 's' (hex_size)
        hex_area = (3 * math.sqrt(3) / 2) * (mesh.hex_size ** 2)

        # Vectores normales precalculados
        thetas = np.array([cell.theta for cell in cells_for_geom])
        normals = np.array([np.cos(thetas), np.sin(thetas)]).T

        # NO guardamos las celdas en caché, solo la geometría estática
        _CACHED_GEOMETRY[cache_key] = (hex_area, normals)

    hex_area, normals = _CACHED_GEOMETRY[cache_key]

    # Obtenemos las celdas ACTUALES de la malla pasada como argumento
    current_cells = mesh.get_all_cells()

    # Extraemos q_vectors de las celdas actuales
    # IMPORTANTE: El orden de current_cells debe coincidir con el orden usado para normals
    # Dado que mesh.get_all_cells() es determinista para una topología dada, esto es válido.
    q_vectors = np.array([cell.q_vector for cell in current_cells])

    # Flujo vectorizado: suma( (q * n) ) * area
    # axis=1 realiza el producto punto fila a fila
    dot_products = np.sum(q_vectors * normals, axis=1)

    total_flux = np.sum(dot_products) * hex_area

    logger.debug("Flujo físico calculado: %.3f", total_flux)
    return float(total_flux)


def fetch_and_apply_torus_field() -> None:
    """Obtiene el campo vectorial de la ECU y lo aplica a la malla global.

    Realiza una solicitud HTTP GET al endpoint `/api/ecu/field_vector` de
    `MATRIZ_ECU_BASE_URL` para obtener el campo vectorial actual. Si la
    solicitud es exitosa y los datos son válidos, llama a
    `apply_external_field_to_mesh` para aplicar este campo a la instancia
    global `malla_cilindrica_global`.

    Maneja errores de red, timeouts y respuestas inválidas. No recibe
    argumentos ni retorna valores directamente, pero puede modificar
    `malla_cilindrica_global` a través de `apply_external_field_to_mesh`.
    """
    ecu_vector_field_url = f"{MATRIZ_ECU_BASE_URL}/api/ecu/field_vector"
    logger.debug(
        "Intentando obtener campo vectorial de ECU en %s", ecu_vector_field_url
    )
    try:
        response = requests.get(ecu_vector_field_url, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
        response_data_dict = response.json()

        if (
            isinstance(response_data_dict, dict)
            and response_data_dict.get("status") == "success"
            and "field_vector" in response_data_dict
        ):
            field_vector = response_data_dict["field_vector"]
            mesh = malla_cilindrica_global
            if mesh:
                if mesh.cells:
                    preview_del_primer_elemento = (
                        field_vector[0] if field_vector else "vacío"
                    )
                    logger.debug(
                        "Aplicando campo vectorial ECU (primer elemento: %s)",
                        preview_del_primer_elemento,
                    )
                    apply_external_field_to_mesh(mesh, field_vector)
                    logger.info(
                        "Campo vectorial toroidal (V) obtenido y aplicado exitosamente."
                    )
                else:
                    logger.warning(
                        "Malla global no tiene celdas. No se aplicará el "
                        "campo vectorial obtenido."
                    )
            else:
                logger.warning(
                    "Malla global no inicializada. No se pudo aplicar el "
                    "campo vectorial obtenido."
                )
        else:
            logger.error(
                "Respuesta JSON inválida, no exitosa, o 'field_vector' "
                "ausente de %s. Respuesta: %s",
                ecu_vector_field_url,
                response_data_dict,
            )

    except requests.exceptions.Timeout:
        logger.error(
            "Timeout al intentar obtener campo vectorial de %s", ecu_vector_field_url
        )
    except requests.exceptions.RequestException as e:
        logger.error(
            "Error de red o HTTP al obtener campo vectorial de %s: %s",
            ecu_vector_field_url,
            e,
        )
    except (ValueError, TypeError, json.JSONDecodeError) as err:
        logger.error(
            "Error al procesar/decodificar respuesta JSON de %s: %s",
            ecu_vector_field_url,
            err,
        )
    except Exception:
        logger.exception(
            "Error inesperado al obtener/aplicar campo vectorial de %s",
            ecu_vector_field_url,
        )


def map_cylinder_to_torus_coords(cell: Cell) -> Tuple[int, int, int]:
    """
    Mapea las coordenadas de una celda del cilindro a coordenadas del toroide.

    Transforma la posición (`theta`, `z`) y el estado de actividad de una celda
    de la malla cilíndrica (`malla_cilindrica_global`) a un sistema de
    coordenadas tridimensional (capa, fila, columna) correspondiente a una
    estructura toroidal definida por `TORUS_NUM_CAPAS`, `TORUS_NUM_FILAS` y
    `TORUS_NUM_COLUMNAS`.

    La coordenada `theta` de la celda se mapea a la columna del toroide.
    La coordenada `z` de la celda se mapea a la fila del toroide.
    La magnitud de la actividad de la celda (combinación de amplitud y
    velocidad) se normaliza y se mapea a la capa del toroide, donde
    mayor actividad corresponde a capas con menor índice (más internas).

    Args:
        cell (Cell): La celda de la malla cilíndrica cuyas coordenadas y estado
            se van a mapear.

    Returns:
        Tuple[int, int, int]: Una tupla `(capa, fila, columna)` con las
            coordenadas mapeadas en el toroide.

    Raises:
        ValueError: Si la malla no está inicializada o las dimensiones del
            toroide son inválidas.
    """
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        raise ValueError("Malla no inicializada o vacía.")

    if TORUS_NUM_CAPAS <= 0 or TORUS_NUM_FILAS <= 0 or TORUS_NUM_COLUMNAS <= 0:
        raise ValueError("Dimensiones del toroide inválidas para mapeo.")

    # Mapeo de theta a columna del toroide (0 a 2*pi -> 0 a num_cols)
    col_float = (cell.theta / (2 * math.pi)) * TORUS_NUM_COLUMNAS
    col = int(math.floor(col_float)) % TORUS_NUM_COLUMNAS  # Periodicidad

    # Mapeo de Z a fila del toroide (min_z a max_z -> 0 a num_rows-1)
    cylinder_height = mesh.max_z - mesh.min_z
    row = 0
    if cylinder_height > EPSILON:
        normalized_z = (cell.z - mesh.min_z) / cylinder_height
        row_float = normalized_z * (TORUS_NUM_FILAS - 1)
        row = int(round(row_float))
        row = max(0, min(row, TORUS_NUM_FILAS - 1))
    elif TORUS_NUM_FILAS > 0:
        row = 0
    else:
        # Esto es un caso de configuración inválida.
        raise ValueError("TORUS_NUM_FILAS debe ser > 0 si la altura del cilindro es 0.")

    # Usar la MAGNITUD de la actividad para mapear a la capa
    activity_magnitude = math.sqrt(cell.amplitude**2 + cell.velocity**2)
    normalized_activity = min(
        1.0, max(0.0, activity_magnitude / MAX_AMPLITUDE_FOR_NORMALIZATION)
    )

    # Mayor actividad -> menor índice de capa (capas "más internas")
    capa = int(round((1.0 - normalized_activity) * (TORUS_NUM_CAPAS - 1)))
    capa = max(0, min(capa, TORUS_NUM_CAPAS - 1))

    return capa, row, col


def send_influence_to_torus(dphi_dt: float) -> None:
    """Envía una influencia a la ECU basada en la tasa de cambio del flujo.

    Construye un vector de influencia donde la primera componente es `dphi_dt`
    y la segunda es cero. Este vector se envía mediante una solicitud HTTP POST
    al endpoint `/api/ecu/influence` de `MATRIZ_ECU_BASE_URL`. La influencia
    se aplica a una ubicación predefinida en el toroide (capa más interna,
    fila y columna centrales).

    Args:
        dphi_dt (float): La tasa de cambio del flujo calculado, que se
            utilizará como magnitud principal de la influencia.

    Maneja errores de red y timeouts. No retorna valores.
    """
    # Definir una ubicación fija o representativa en el toroide
    target_capa = 0  # Capa más interna/crítica
    target_row = TORUS_NUM_FILAS // 2  # Fila central
    target_col = TORUS_NUM_COLUMNAS // 2  # Columna central

    influence_vector = [dphi_dt, 0.0]
    watcher_name = f"malla_watcher_dPhiDt{dphi_dt:.3f}"

    payload = {
        KEY_CAPA: target_capa,
        KEY_ROW: target_row,
        KEY_COL: target_col,
        KEY_VECTOR: influence_vector,
        KEY_NOMBRE_WATCHER: watcher_name,
    }

    ecu_influence_url = f"{MATRIZ_ECU_BASE_URL}/api/ecu/influence"
    try:
        response = requests.post(
            ecu_influence_url, json=payload, timeout=REQUESTS_TIMEOUT
        )
        response.raise_for_status()
        logger.info(
            "Influencia dPhi/dt a %s (%d,%d,%d). Payload: %s. Status: %d",
            ecu_influence_url,
            target_capa,
            target_row,
            target_col,
            payload,
            response.status_code,
        )
    except requests.exceptions.Timeout:
        logger.error("Timeout al enviar influencia (dPhi/dt) a %s", ecu_influence_url)
    except requests.exceptions.RequestException as e:
        logger.error(
            "Error de red al enviar influencia (dPhi/dt) a %s: %s", ecu_influence_url, e
        )
    except Exception:
        logger.exception(
            "Error inesperado al enviar influencia (dPhi/dt) a %s", ecu_influence_url
        )


# --- Bucle de Simulación (Thread) ---
simulation_thread = None
stop_simulation_event = threading.Event()


def _calculate_flux_change(mesh: HexCylindricalMesh, dt: float) -> float:
    """Calcula el flujo y su tasa de cambio (dPhi/dt)."""
    current_flux = calculate_flux(mesh)
    dphi_dt = (current_flux - mesh.previous_flux) / dt if dt > 0 else 0.0
    mesh.previous_flux = current_flux
    logger.debug("Flujo actual=%.3f, dPhi/dt=%.3f", current_flux, dphi_dt)
    return dphi_dt


def _send_influence_if_needed(dphi_dt: float):
    """Comprueba si dPhi/dt supera el umbral y envía influencia si es así."""
    if abs(dphi_dt) > DPHI_DT_INFLUENCE_THRESHOLD:
        logger.info(
            "|dPhi/dt|=%.3f supera umbral %.3f. Enviando influencia...",
            abs(dphi_dt),
            DPHI_DT_INFLUENCE_THRESHOLD,
        )
        send_influence_to_torus(dphi_dt)
    else:
        logger.debug(
            "|dPhi/dt|=%.3f no supera umbral para influenciar toroide.", abs(dphi_dt)
        )


def simulation_loop() -> None:
    """Ejecuta el bucle principal de simulación de la malla.

    Este bucle se ejecuta en un hilo separado y orquesta las operaciones en
    cada paso de la simulación:
    1. Obtiene y aplica el campo externo de la ECU.
    2. Calcula la tasa de cambio del flujo (dPhi/dt).
    3. Simula la dinámica interna de la malla (osciladores acoplados).
    4. Actualiza las métricas de estado agregado de la malla.
    5. Envía una influencia a la ECU si el cambio en el flujo es significativo.
    """
    logger.info("Iniciando bucle de simulación de malla...")
    step_count = 0
    dt = SIMULATION_INTERVAL

    mesh = malla_cilindrica_global
    if mesh is None:
        logger.error(
            "Malla global no inicializada. No se puede iniciar el bucle de simulación."
        )
        return

    if not hasattr(mesh, "previous_flux"):
        mesh.previous_flux = 0.0
        logger.warning(
            "previous_flux no inicializado en malla_cilindrica_global. "
            "Inicializando a 0.0"
        )

    while not stop_simulation_event.is_set():
        start_time = time.monotonic()
        step_count += 1
        logger.debug("--- Iniciando paso de simulación %d ---", step_count)

        try:
            fetch_and_apply_torus_field()
            dphi_dt = _calculate_flux_change(mesh, dt)
            simular_paso_malla()
            update_aggregate_state()
            _send_influence_if_needed(dphi_dt)

        except Exception:
            logger.exception(
                "Error durante el paso de simulación %d de malla.", step_count
            )

        elapsed_time = time.monotonic() - start_time
        sleep_time = max(0, dt - elapsed_time)
        logger.debug(
            "--- Paso %d completado en %.3fs. Durmiendo por %.3fs ---",
            step_count,
            elapsed_time,
            sleep_time,
        )
        if sleep_time > 0:
            stop_simulation_event.wait(sleep_time)

    logger.info("Bucle de simulación de malla detenido.")


AGENT_AI_REGISTER_URL = os.environ.get(
    "AGENT_AI_REGISTER_URL", "http://agent_ai:9000/api/register"
)
MAX_REGISTRATION_RETRIES = 5
RETRY_DELAY = 5


def register_with_agent_ai(
    module_name: str,
    module_url: str,
    health_url: str,
    module_type: str,
    aporta_a: str,
    naturaleza: str,
    description: str = "",
) -> bool:
    """Intenta registrar este módulo con el servicio AgentAI.

    Realiza una solicitud HTTP POST al endpoint `/api/register` de
    `AGENT_AI_REGISTER_URL` para registrar este módulo (malla_watcher).
    El payload incluye el nombre, URL del módulo, URL de salud, tipo,
    a qué componente principal aporta y su naturaleza.

    Realiza hasta `MAX_REGISTRATION_RETRIES` intentos en caso de fallo,
    con un retraso de `RETRY_DELAY` segundos entre intentos.

    Args:
        module_name (str): El nombre del módulo a registrar.
        module_url (str): La URL base del módulo.
        health_url (str): La URL del endpoint de salud del módulo.
        module_type (str): El tipo de módulo (ej. "auxiliar").
        aporta_a (str): El nombre del componente principal al que este
            módulo aporta (ej. "matriz_ecu").
        naturaleza (str): La naturaleza de la contribución del módulo
            (ej. "modulador").
        description (str, optional): Una descripción breve del módulo.
            Defaults to "".

    Returns:
        bool:
        True si el registro fue exitoso (HTTP 200), False en caso contrario
        después de todos los reintentos.
    """
    payload = {
        KEY_NOMBRE: module_name,
        KEY_URL: module_url,
        KEY_URL_SALUD: health_url,
        KEY_TIPO: module_type,
        KEY_APORTA_A: aporta_a,
        KEY_NATURALEZA_AUXILIAR: naturaleza,
        KEY_DESCRIPCION: description,
    }
    logger.info(
        "Intentando registrar '%s' en AgentAI (%s)...",
        module_name,
        AGENT_AI_REGISTER_URL,
    )
    for attempt in range(MAX_REGISTRATION_RETRIES):
        try:
            response = requests.post(AGENT_AI_REGISTER_URL, json=payload, timeout=4.0)
            response.raise_for_status()
            if response.status_code == 200:
                logger.info("Registro de '%s' exitoso en AgentAI.", module_name)
                return True
            else:
                logger.warning(
                    "Registro de '%s' recibido con status %d. Respuesta: %s",
                    module_name,
                    response.status_code,
                    response.text,
                )
        except requests.exceptions.RequestException as e:
            logger.error(
                "Error de conexión al intentar registrar '%s' (intento %d/%d): %s",
                module_name,
                attempt + 1,
                MAX_REGISTRATION_RETRIES,
                e,
            )
        except Exception as err_info:
            logger.error(
                "Error inesperado durante el registro de '%s' (intento %d/%d): %s",
                module_name,
                attempt + 1,
                MAX_REGISTRATION_RETRIES,
                err_info,
            )

        if attempt < MAX_REGISTRATION_RETRIES - 1:
            logger.info("Reintentando registro en %d segundos...", RETRY_DELAY)
            time.sleep(RETRY_DELAY)
        else:
            logger.error(
                "No se pudo registrar '%s' en AgentAI después de %d intentos.",
                module_name,
                MAX_REGISTRATION_RETRIES,
            )
            return False
    return False


# --- Servidor Flask ---
app = Flask(__name__)


@app.errorhandler(Exception)
def handle_global_exception(e: Exception) -> Tuple[str, int]:
    """
    Manejador de excepciones global para la aplicación Flask.
    Captura cualquier excepción no manejada, la registra y devuelve una
    respuesta JSON de error genérica.
    """
    if isinstance(e, HTTPException):
        # Para excepciones HTTP, usar sus descripciones y códigos estándar
        response = e.get_response()
        return jsonify(
            {
                KEY_STATUS: KEY_ERROR,
                KEY_MESSAGE: e.description,
            }
        ), response.status_code

    # Para cualquier otra excepción, es un error 500 del servidor
    logger.exception("Error no manejado en una solicitud de API: %s", e)
    return jsonify(
        {KEY_STATUS: KEY_ERROR, KEY_MESSAGE: "Ocurrió un error interno en el servidor."}
    ), 500


# --- Endpoints de Flask ---
@app.route("/api/health", methods=["GET"])
def health_check() -> Tuple[str, int]:
    """Verifica el estado de salud del servicio Malla Watcher.

    Comprueba varios aspectos:
    - Si la instancia de `HexCylindricalMesh` global está inicializada.
    - Si la malla contiene celdas.
    - Si el hilo de simulación del resonador está activo.
    - La conectividad estructural de la malla (número de vecinos por celda).

    Retorna un JSON con el estado general ("success", "warning", "error"),
    un mensaje descriptivo y detalles específicos sobre cada componente
    verificado. El código de estado HTTP refleja el estado general.

    Returns:
        Tuple[str, int]: Una tupla conteniendo la respuesta JSON como string
                         y el código de estado HTTP.
    """
    sim_alive = simulation_thread.is_alive() if simulation_thread else False
    mesh = malla_cilindrica_global
    num_cells = len(mesh.cells) if mesh and mesh.cells else 0
    status = KEY_SUCCESS
    message = "Malla_watcher operativo."

    connectivity_status = "N/A"
    min_neighbors = -1
    max_neighbors = -1

    if mesh is None:
        status = KEY_ERROR
        message = "Error: Objeto HexCylindricalMesh global no inicializado."
    elif num_cells == 0:
        status = KEY_ERROR
        message = "Error: Malla inicializada pero contiene 0 celdas."
    elif not sim_alive:
        status = KEY_ERROR
        message = "Error: Hilo de simulación del resonador inactivo."
    else:
        try:
            connectivity_counts = mesh.verify_connectivity()
            if connectivity_counts:
                min_neighbors = min(connectivity_counts.keys())
                max_neighbors = max(connectivity_counts.keys())
                if max_neighbors > 6 or (min_neighbors < 3 and num_cells > 1):
                    connectivity_status = "error"
                    status = KEY_ERROR
                    message = "Error estructural: Problemas graves de conectividad."
                elif min_neighbors < 6 and num_cells > 1:
                    connectivity_status = "warning"
                    if status == KEY_SUCCESS:
                        status = KEY_WARNING
                        message = "Advertencia: Posibles problemas de conectividad."
                else:
                    connectivity_status = "ok"
            else:
                connectivity_status = "warning"
                if status == KEY_SUCCESS:
                    status = KEY_WARNING
                    message = "Advertencia: verify_connectivity retornó vacío."
        except Exception as err_info:
            logger.error("Error durante verify_connectivity: %s", err_info)
            connectivity_status = "error"
            status = KEY_ERROR
            message = f"Error interno al verificar conectividad: {err_info}"

    # --- Construir Respuesta JSON Detallada ---
    response_data = {
        KEY_STATUS: status,
        "module": "Malla_watcher",
        KEY_MESSAGE: message,
        KEY_DETAILS: {
            "mesh": {
                "initialized": mesh is not None,
                "num_cells": num_cells,
                "connectivity_status": connectivity_status,
                "min_neighbors": min_neighbors,
                "max_neighbors": max_neighbors,
                "z_periodic": mesh.periodic_z if mesh else None,
            },
            "resonator_simulation": {
                "running": sim_alive,
            },
        },
    }

    http_status_code = 200
    if status == KEY_WARNING:
        http_status_code = 503
    elif status == KEY_ERROR:
        http_status_code = 500

    logger.debug("Health check response: Status=%s, HTTP=%d", status, http_status_code)
    return jsonify(response_data), http_status_code


@app.route("/api/state", methods=["GET"])
def get_malla_state() -> Tuple[str, int]:
    """
    Devuelve el estado agregado actual de la malla y parámetros de control.

    Retorna un JSON que incluye:
    - Las métricas de estado agregado de la malla.
    - Los parámetros de control actuales (coeficientes C de PhosWave y D de
      Electron).
    - El número total de celdas en la malla.

    Returns:
        Tuple[str, int]: Una tupla conteniendo la respuesta JSON como string
                         con el estado y el código de estado HTTP 200.
    """
    state_data = {}
    with aggregate_state_lock:
        state_data.update(aggregate_state)

    with control_lock:
        state_data["control_params"] = {
            KEY_PHOSWAVE_C: resonador_global.C,
            KEY_ELECTRON_D: electron_global.D,
        }

    mesh = malla_cilindrica_global
    state_data["num_cells"] = len(mesh.cells) if mesh and mesh.cells else 0

    logger.debug("Devolviendo estado agregado: %s", state_data)
    return jsonify({KEY_STATUS: KEY_SUCCESS, "state": state_data})


@app.route("/api/control", methods=["POST"])
def set_malla_control() -> Tuple[str, int]:
    """
    Ajusta parámetros de control de la malla (acoplamiento y amortiguación).

    Espera un payload JSON con una clave "control_signal" (numérica).
    Esta señal se utiliza para modular los coeficientes base de acoplamiento
    (`BASE_COUPLING_T`) y amortiguación (`BASE_DAMPING_E`) mediante ganancias
    (`K_GAIN_COUPLING`, `K_GAIN_DAMPING`).

    Los nuevos coeficientes calculados se aplican a las instancias globales
    `resonador_global` (PhosWave) y `electron_global` (Electron).

    Retorna un JSON indicando el éxito o fracaso de la operación y los
    nuevos parámetros de control.

    Returns:
        Tuple[str, int]:
        Una tupla conteniendo la respuesta JSON como string
        y el código de estado HTTP (200 si éxito, 400 si
        error en la solicitud).
    """
    data = request.get_json(silent=True)
    if data is None or "control_signal" not in data:
        logger.error(
            "Solicitud a /api/control sin payload JSON válido o 'control_signal'"
        )
        return jsonify(
            {
                KEY_STATUS: KEY_ERROR,
                KEY_MESSAGE: "Payload JSON vacío, inválido o falta 'control_signal'",
            }
        ), 400

    signal = data["control_signal"]
    if not isinstance(signal, (int, float)):
        return jsonify(
            {
                KEY_STATUS: KEY_ERROR,
                KEY_MESSAGE: "El campo 'control_signal' debe ser un número.",
            }
        ), 400

    with control_lock:
        new_C = max(0.0, BASE_COUPLING_T + K_GAIN_COUPLING * signal)
        new_D = max(0.0, BASE_DAMPING_E - K_GAIN_DAMPING * signal)
        resonador_global.ajustar_coeficientes(new_C)
        electron_global.ajustar_coeficientes(new_D)
        control_params[KEY_PHOSWAVE_C] = resonador_global.C
        control_params[KEY_ELECTRON_D] = electron_global.D

    logger.info(
        "Parámetros de control ajustados: C=%.3f, D=%.3f (señal=%.3f)",
        resonador_global.C,
        electron_global.D,
        signal,
    )
    return jsonify(
        {
            KEY_STATUS: KEY_SUCCESS,
            KEY_MESSAGE: "Parámetros ajustados",
            "current_params": control_params,
        }
    ), 200


@app.route("/api/malla", methods=["GET"])
def get_malla() -> Tuple[str, int]:
    """
    Devuelve la estructura y estado detallado de todas las celdas de la malla.

    Retorna un JSON que contiene:
    - Metadatos de la malla: radio, número de celdas, periodicidad en Z,
      límites en Z.
    - Una lista de todas las celdas, donde cada celda es un diccionario
      obtenido a través de su método `to_dict()`.

    Si la malla no está inicializada o está vacía, retorna un error 503.

    Returns:
        Tuple[str, int]:
        Una tupla conteniendo la respuesta JSON como string
        y el código de estado HTTP (200 si éxito, 503 si
        la malla no está disponible).
    """
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        return jsonify(
            {KEY_STATUS: KEY_ERROR, KEY_MESSAGE: "Malla no inicializada o vacía."}
        ), 503

    return jsonify(
        {
            KEY_STATUS: KEY_SUCCESS,
            "metadata": {
                "radius": mesh.radius,
                "num_cells": len(mesh.cells),
                "periodic_z": mesh.periodic_z,
                "z_bounds": {"min": mesh.min_z, "max": mesh.max_z},
            },
            "cells": [cell.to_dict() for cell in mesh.get_all_cells()],
        }
    ), 200


@app.route("/api/malla/influence", methods=["POST"])
def aplicar_influencia_toroide_push() -> Tuple[str, int]:
    """Aplica un campo vectorial externo (push) a la malla.

    Este endpoint es una alternativa pasiva a `fetch_and_apply_torus_field`.
    Recibe un campo vectorial completo, típicamente de la ECU, a través de
    una solicitud HTTP POST. El payload JSON esperado debe contener la clave
    `"field_vector"` con el campo (estructura anidada de listas que se
    convierte a un array NumPy de shape [capas, filas, columnas, 2]).

    El campo recibido se aplica a la instancia global `malla_cilindrica_global`
    llamando a `apply_external_field_to_mesh`, que actualiza los `q_vector`
    de las celdas mediante interpolación bilineal.

    Retorna un JSON indicando el resultado de la operación.

    Returns:
        Tuple[str, int]: Una tupla conteniendo la respuesta JSON como string
                         y el código de estado HTTP (200 si éxito, 400 si
                         el payload es inválido, 500 si hay error interno,
                         503 si la malla no está disponible).
    """
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        logger.warning(
            "Malla no inicializada o vacía. Saltando aplicación de influencia."
        )
        return jsonify(
            {
                KEY_STATUS: KEY_WARNING,
                KEY_MESSAGE: "Malla no inicializada o vacía. Influencia no aplicada.",
            }
        ), 503

    logger.warning(
        "Recibida llamada a endpoint pasivo /api/malla/influence. "
        "Se recomienda usar el fetch activo."
    )
    data = request.get_json(silent=True)
    if data is None or "field_vector" not in data:
        logger.error("Solicitud POST a /api/malla/influence sin 'field_vector'.")
        return jsonify(
            {
                KEY_STATUS: KEY_ERROR,
                KEY_MESSAGE: "Payload JSON vacío, inválido o falta 'field_vector'",
            }
        ), 400

    field_vector_data = data["field_vector"]

    try:
        apply_external_field_to_mesh(mesh, field_vector_data)
        logger.info(
            "Influencia del campo vectorial toroidal (push) aplicadacorrectamente."
        )
        return jsonify(
            {
                KEY_STATUS: KEY_SUCCESS,
                KEY_MESSAGE: "CV externo aplicado a q_vector de las celdas.",
            }
        ), 200

    except ValueError as ve:
        logger.error(
            "Error de valor al procesar campo vectorial externo (push): %s", ve
        )
        return jsonify(
            {
                KEY_STATUS: KEY_ERROR,
                KEY_MESSAGE: f"Error en los datos recibidos (push): {ve}",
            }
        ), 400
    except Exception:
        logger.exception("Error inesperado al aplicar influencia del toroide.")
        return jsonify(
            {
                KEY_STATUS: KEY_ERROR,
                KEY_MESSAGE: "Error interno al aplicar campo externo (push).",
            }
        ), 500


@app.route("/api/event", methods=["POST"])
def receive_event() -> Tuple[str, int]:
    """
    Procesa un evento externo y lo aplica a una celda específica de la malla.

    Actualmente, solo soporta eventos de tipo "pulse".
    Espera un payload JSON con:
    - `"type"`: "pulse"
    - `"coords"`: {"q": int, "r": int} (coordenadas axiales de la celda)
    - `"magnitude"`: float (magnitud del pulso a aplicar a la velocidad
      de la celda)

    Si la celda especificada existe, su velocidad se incrementa por la
    magnitud del pulso.

    Retorna un JSON indicando el resultado de la operación.

    Returns:
        Tuple[str, int]:
        Una tupla conteniendo la respuesta JSON como string
        y el código de estado HTTP (200 si éxito, 400 si
        el payload es inválido o datos incompletos, 404 si
        la celda no se encuentra, 503 si la malla no está
        disponible).
    """
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        logger.warning("Malla no inicializada o vacía. Saltando aplicación de evento.")
        return jsonify(
            {
                KEY_STATUS: KEY_WARNING,
                KEY_MESSAGE: "Malla no inicializada o vacía. Evento no aplicado.",
            }
        ), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify(
            {KEY_STATUS: KEY_ERROR, KEY_MESSAGE: "Payload JSON vacío o inválido"}
        ), 400

    logger.info("Evento recibido: %s", data)
    if data.get("type") == "pulse" and "coords" in data and "magnitude" in data:
        q_coord = data["coords"].get("q")
        r_coord = data["coords"].get("r")
        try:
            mag = float(data.get("magnitude", 1.0))
        except (ValueError, TypeError):
            return jsonify(
                {KEY_STATUS: KEY_ERROR, KEY_MESSAGE: "Magnitud inválida"}
            ), 400

        if q_coord is not None and r_coord is not None:
            cell = mesh.get_cell(q_coord, r_coord)
            if cell:
                logger.info(
                    "Celda (%d,%d) - Velocidad antes: %.3f",
                    q_coord,
                    r_coord,
                    cell.velocity,
                )
                cell.velocity += mag
                logger.info(
                    "Celda (%d,%d) - Velocidad después: %.3f",
                    q_coord,
                    r_coord,
                    cell.velocity,
                )
            else:
                logger.warning(
                    "No se encontró celda (%d,%d) para aplicar evento.",
                    q_coord,
                    r_coord,
                )
                return jsonify(
                    {
                        KEY_STATUS: KEY_WARNING,
                        KEY_MESSAGE: f"Celda ({q_coord},{r_coord}) no encontrada.",
                    }
                ), 404
        else:
            return jsonify(
                {
                    KEY_STATUS: KEY_ERROR,
                    KEY_MESSAGE: "Coordenadas 'q' o 'r' faltantes o inválidas.",
                }
            ), 400
    else:
        return jsonify(
            {
                KEY_STATUS: KEY_ERROR,
                KEY_MESSAGE: "Tipo de evento no soportado o datos incompletos.",
            }
        ), 400

    return jsonify({KEY_STATUS: KEY_SUCCESS, KEY_MESSAGE: "Evento procesado"}), 200


@app.route("/api/error", methods=["POST"])
def receive_error() -> Tuple[str, int]:
    """Recibe y registra un mensaje de error reportado por otro servicio.

    Espera un payload JSON que contenga los detalles del error.
    Este endpoint simplemente registra la información recibida como un error
    en los logs del Malla Watcher.

    Args:
        No toma argumentos directos de la función, pero espera un JSON en el
        cuerpo de la solicitud POST.

    Returns:
        Tuple[str, int]:
        Una tupla conteniendo una respuesta JSON de confirmación
        y el código de estado HTTP (200 si el payload es
        procesado, 400 si el payload JSON es inválido o vacío).
    """
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(
            {KEY_STATUS: KEY_ERROR, KEY_MESSAGE: "Payload JSON vacío o inválido"}
        ), 400

    logger.error("Error reportado desde otro servicio: %s", data)
    return jsonify({KEY_STATUS: KEY_SUCCESS, KEY_MESSAGE: "Error procesado"}), 200


@app.route("/api/config", methods=["GET"])
def get_config() -> Tuple[str, int]:
    """Devuelve la configuración actual del Malla Watcher.

    Retorna un JSON que incluye detalles sobre:
    - Configuración de la malla (radio, segmentos, tamaño de hexágono, etc.).
    - Configuración de comunicación (URL de ECU, dimensiones del toroide,
      umbrales).
    - Configuración de simulación (intervalo, umbral dPhi/dt).
    - Configuración de control (parámetros base y actuales de acoplamiento
      y amortiguación).

    Returns:
        Tuple[str, int]: Una tupla conteniendo la respuesta JSON con la
                         configuración y el código de estado HTTP 200.
    """
    mesh = malla_cilindrica_global
    return jsonify(
        {
            KEY_STATUS: KEY_SUCCESS,
            "config": {
                "malla_config": {
                    "radius": float(os.environ.get("MW_RADIUS", 5.0)),
                    "height_segments": int(os.environ.get("MW_HEIGHT_SEG", 6)),
                    "circumference_segments_target": int(
                        os.environ.get("MW_CIRCUM_SEG", 12)
                    ),
                    "circumference_segments_actual": mesh.circumference_segments_actual
                    if mesh
                    else None,
                    "hex_size": float(os.environ.get("MW_HEX_SIZE", 1.0)),
                    "periodic_z": os.environ.get("MW_PERIODIC_Z", "True").lower()
                    == "true",
                },
                "communication_config": {
                    "matriz_ecu_url": MATRIZ_ECU_BASE_URL,
                    # SOLUCIÓN E501: Se divide el f-string para que la línea
                    # sea más corta y legible.
                    "torus_dims": (
                        f"{TORUS_NUM_CAPAS}x{TORUS_NUM_FILAS}x{TORUS_NUM_COLUMNAS}"
                    ),
                    "influence_threshold": AMPLITUDE_INFLUENCE_THRESHOLD,
                    "max_activity_normalization": MAX_AMPLITUDE_FOR_NORMALIZATION,
                },
                "simulation_config": {
                    "interval": SIMULATION_INTERVAL,
                    "dphi_dt_influence_threshold": DPHI_DT_INFLUENCE_THRESHOLD,
                },
                "control_config": {
                    "base_coupling_t": BASE_COUPLING_T,
                    "base_damping_e": BASE_DAMPING_E,
                    "k_gain_coupling": K_GAIN_COUPLING,
                    "k_gain_damping": K_GAIN_DAMPING,
                    "current_coupling_C": resonador_global.C,
                    "current_damping_D": electron_global.D,
                },
            },
        }
    ), 200


# --- Punto de Entrada Principal ---


def main():
    """Función principal para inicializar y ejecutar el servicio."""
    global malla_cilindrica_global
    global simulation_thread
    global resonador_global
    global electron_global
    global MATRIZ_ECU_BASE_URL
    global TORUS_NUM_CAPAS
    global TORUS_NUM_FILAS
    global TORUS_NUM_COLUMNAS
    global AMPLITUDE_INFLUENCE_THRESHOLD
    global MAX_AMPLITUDE_FOR_NORMALIZATION
    global REQUESTS_TIMEOUT
    global BASE_COUPLING_T
    global BASE_DAMPING_E
    global K_GAIN_COUPLING
    global K_GAIN_DAMPING
    global SIMULATION_INTERVAL
    global DPHI_DT_INFLUENCE_THRESHOLD

    # 1. Leer las variables de configuración REALES desde el entorno
    MESH_RADIUS_REAL = float(os.environ.get("MW_RADIUS", 5.0))
    MESH_HEIGHT_SEGMENTS_REAL = int(os.environ.get("MW_HEIGHT_SEG", 6))
    MESH_CIRCUMFERENCE_SEGMENTS_REAL = int(os.environ.get("MW_CIRCUM_SEG", 12))
    MESH_HEX_SIZE_REAL = float(os.environ.get("MW_HEX_SIZE", 1.0))
    MESH_PERIODIC_Z_REAL = os.environ.get("MW_PERIODIC_Z", "True").lower() == "true"

    MATRIZ_ECU_BASE_URL_REAL = os.environ.get("MATRIZ_ECU_URL", "http://ecu:8000")
    TORUS_NUM_CAPAS_REAL = int(os.environ.get("TORUS_NUM_CAPAS", 3))
    TORUS_NUM_FILAS_REAL = int(os.environ.get("TORUS_NUM_FILAS", 4))
    TORUS_NUM_COLUMNAS_REAL = int(os.environ.get("TORUS_NUM_COLUMNAS", 5))
    AMPLITUDE_INFLUENCE_THRESHOLD_REAL = float(
        os.environ.get("MW_INFLUENCE_THRESHOLD", 5.0)
    )
    MAX_AMPLITUDE_FOR_NORMALIZATION_REAL = float(
        os.environ.get("MW_MAX_AMPLITUDE_NORM", 20.0)
    )
    REQUESTS_TIMEOUT_REAL = float(os.environ.get("MW_REQUESTS_TIMEOUT", 2.0))
    BASE_COUPLING_T_REAL = float(os.environ.get("MW_BASE_T", 0.6))
    BASE_DAMPING_E_REAL = float(os.environ.get("MW_BASE_E", 0.1))
    K_GAIN_COUPLING_REAL = float(os.environ.get("MW_K_GAIN_T", 0.1))
    K_GAIN_DAMPING_REAL = float(os.environ.get("MW_K_GAIN_E", 0.05))
    SIMULATION_INTERVAL_REAL = float(os.environ.get("MW_SIM_INTERVAL", 0.5))
    DPHI_DT_INFLUENCE_THRESHOLD_REAL = float(
        os.environ.get("MW_DPHI_DT_THRESHOLD", 1.0)
    )

    # 2. Loguear la configuración REAL
    logger.info(
        "Configuración Malla REAL: R=%.1f, HSeg=%d, CSeg=%d, Hex=%.1f, PerZ=%s",
        MESH_RADIUS_REAL,
        MESH_HEIGHT_SEGMENTS_REAL,
        MESH_CIRCUMFERENCE_SEGMENTS_REAL,
        MESH_HEX_SIZE_REAL,
        MESH_PERIODIC_Z_REAL,
    )
    logger.info(
        "Configuración Comms REAL: ECU_URL=%s, Torus=%dx%dx%d, InfThr=%.1f",
        MATRIZ_ECU_BASE_URL_REAL,
        TORUS_NUM_CAPAS_REAL,
        TORUS_NUM_FILAS_REAL,
        TORUS_NUM_COLUMNAS_REAL,
        AMPLITUDE_INFLUENCE_THRESHOLD_REAL,
    )
    logger.info(
        "Configuración Control REAL: BaseC=%.1f, BaseD=%.1f, GainC=%.1f, GainD=%.1f",
        BASE_COUPLING_T_REAL,
        BASE_DAMPING_E_REAL,
        K_GAIN_COUPLING_REAL,
        K_GAIN_DAMPING_REAL,
    )
    logger.info(
        "Configuración Simulación REAL: Interval=%.1fs, dPhi/dt Threshold=%.1f",
        SIMULATION_INTERVAL_REAL,
        DPHI_DT_INFLUENCE_THRESHOLD_REAL,
    )
    logger.info(
        "Configuración Normalización Influencia REAL: MaxActivityNorm=%.1f",
        MAX_AMPLITUDE_FOR_NORMALIZATION_REAL,
    )

    # 3. RE-Inicializar las instancias globales REAL
    MATRIZ_ECU_BASE_URL = MATRIZ_ECU_BASE_URL_REAL
    TORUS_NUM_CAPAS = TORUS_NUM_CAPAS_REAL
    TORUS_NUM_FILAS = TORUS_NUM_FILAS_REAL
    TORUS_NUM_COLUMNAS = TORUS_NUM_COLUMNAS_REAL
    AMPLITUDE_INFLUENCE_THRESHOLD = AMPLITUDE_INFLUENCE_THRESHOLD_REAL
    MAX_AMPLITUDE_FOR_NORMALIZATION = MAX_AMPLITUDE_FOR_NORMALIZATION_REAL
    REQUESTS_TIMEOUT = REQUESTS_TIMEOUT_REAL
    BASE_COUPLING_T = BASE_COUPLING_T_REAL
    BASE_DAMPING_E = BASE_DAMPING_E_REAL
    K_GAIN_COUPLING = K_GAIN_COUPLING_REAL
    K_GAIN_DAMPING = K_GAIN_DAMPING_REAL
    SIMULATION_INTERVAL = SIMULATION_INTERVAL_REAL
    DPHI_DT_INFLUENCE_THRESHOLD = DPHI_DT_INFLUENCE_THRESHOLD_REAL

    try:
        logger.info(
            "Intentando RE-inicializar la instancia global REAL de "
            "HexCylindricalMesh..."
        )
        malla_cilindrica_global = HexCylindricalMesh(
            radius=MESH_RADIUS_REAL,
            height_segments=MESH_HEIGHT_SEGMENTS_REAL,
            circumference_segments_target=MESH_CIRCUMFERENCE_SEGMENTS_REAL,
            hex_size=MESH_HEX_SIZE_REAL,
            periodic_z=MESH_PERIODIC_Z_REAL,
        )
        if not malla_cilindrica_global.cells:
            logger.error("¡La malla REAL se inicializó pero está vacía!")
            exit(1)
        else:
            logger.info(
                "Malla REAL inicializada con %d celdas.",
                len(malla_cilindrica_global.cells),
            )
    except Exception:
        logger.exception(
            "Error crítico al RE-inicializar HexCylindricalMesh global REAL."
        )
        exit(1)

    resonador_global = PhosWave(coef_acoplamiento=BASE_COUPLING_T)
    electron_global = Electron(coef_amortiguacion=BASE_DAMPING_E)

    if not hasattr(malla_cilindrica_global, "previous_flux"):
        malla_cilindrica_global.previous_flux = 0.0
        logger.info("previous_flux inicializado a 0.0 en instancia REAL.")
    logger.debug(
        "previous_flux en instancia REAL: %.1f", malla_cilindrica_global.previous_flux
    )

    with control_lock:
        control_params["phoswave_C"] = resonador_global.C
        control_params["electron_D"] = electron_global.D
    logger.info(
        "Parámetros de control iniciales REALES: C=%.3f, D=%.3f",
        control_params["phoswave_C"],
        control_params["electron_D"],
    )

    # 4. Registro con AgentAI
    MODULE_NAME = "malla_watcher"
    SERVICE_PORT = int(os.environ.get("PORT", 5001))
    MODULE_URL = f"http://{MODULE_NAME}:{SERVICE_PORT}"
    HEALTH_URL = f"{MODULE_URL}/api/health"
    APORTA_A = "matriz_ecu"
    NATURALEZA = "modulador"
    DESCRIPTION = (
        "Simulador de malla hexagonal cilíndrica (osciladores "
        "acoplados) acoplado a ECU, influye basado en inducción "
        "electromagnética."
    )

    registration_successful = register_with_agent_ai(
        MODULE_NAME,
        MODULE_URL,
        HEALTH_URL,
        "auxiliar",
        APORTA_A,
        NATURALEZA,
        DESCRIPTION,
    )
    if not registration_successful:
        logger.warning(
            "El módulo '%s' continuará sin registro exitoso en AgentAI.", MODULE_NAME
        )

    # 5. Iniciar Hilo de Simulación y Servidor Flask
    logger.info("Creando e iniciando hilo de simulación de malla...")
    stop_simulation_event.clear()
    simulation_thread = threading.Thread(
        target=simulation_loop, daemon=True, name="MallaSimLoop"
    )
    simulation_thread.start()
    logger.info("Hilo de simulación iniciado.")

    logger.info("Iniciando servicio Flask de malla_watcher en puerto %d", SERVICE_PORT)
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=False, use_reloader=False)

    # Código de limpieza al detener el servidor
    logger.info("Señal de detención recibida. Deteniendo hilo de simulación.")
    stop_simulation_event.set()
    if simulation_thread:
        simulation_thread.join(timeout=SIMULATION_INTERVAL * 3 + 5)
        if simulation_thread.is_alive():
            logger.warning("El hilo de simulación no terminó a tiempo.")
    logger.info("Servicio malla_watcher finalizado.")


if __name__ == "__main__":
    main()

# [end of watchers/watchers_tools/malla_watcher/malla_watcher.py]
