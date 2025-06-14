#!/usr/bin/env python3
"""
malla_watcher.py

Define una malla hexagonal cilíndrica inspirada en la estructura del
grafeno, modelada como un sistema de osciladores acoplados. La malla
interactúa con un campo vectorial externo (proveniente de ECU) que modula
el acoplamiento entre osciladores, y genera influencias sobre dicho campo
basadas en la tasa de cambio del flujo del campo a través de la malla
(analogía de inducción electromagnética).

Componentes clave:
- **Cell**: Representa un oscilador en la malla con estado (amplitud,
  velocidad) y campo externo local (q_vector).
- **HexCylindricalMesh** (importado desde `cilindro_grafenal`): Gestiona la
  malla hexagonal cilíndrica con validación de conectividad y condiciones
  de contorno periódicas.

Interacciones:
1. **ECU → Malla**: Obtiene periódicamente el campo vectorial de ECU y lo
   aplica a las celdas mediante interpolación.
2. **Malla → ECU**: Envía influencias al toroide basadas en la tasa de
   cambio del flujo del campo de ECU a través de la malla (dPhi/dt).

Dependencias:
- `cilindro_grafenal.HexCylindricalMesh`: Clase central para la generación
  y validación de la estructura digital del cilindro.
"""

import math
import logging
import requests
import time
import threading
import os
import json
from flask import Flask, request, jsonify
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from .utils.cilindro_grafenal import HexCylindricalMesh, Cell

# --- Configuración del Logging ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("malla_watcher")
if not logger.hasHandlers():
    handler = logging.FileHandler(os.path.join(log_dir, "malla_watcher.log"))
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
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
AMPLITUDE_INFLUENCE_THRESHOLD = float(
    os.environ.get("MW_INFLUENCE_THRESHOLD", 5.0))
# Valor para normalizar métrica de actividad
MAX_AMPLITUDE_FOR_NORMALIZATION = float(
    os.environ.get("MW_MAX_AMPLITUDE_NORM", 20.0))
REQUESTS_TIMEOUT = float(os.environ.get("MW_REQUESTS_TIMEOUT", 2.0))

# --- Constantes de Configuración para Control ---
BASE_COUPLING_T = float(os.environ.get("MW_BASE_T", 0.6))
K_GAIN_COUPLING = float(os.environ.get("MW_K_GAIN_T", 0.1))
BASE_DAMPING_E = float(os.environ.get("MW_BASE_E", 0.1))
K_GAIN_DAMPING = float(os.environ.get("MW_K_GAIN_E", 0.05))

# --- Constantes de Configuración para Simulación ---
SIMULATION_INTERVAL = float(
    os.environ.get("MW_SIM_INTERVAL", 0.5))  # Segundos (esto es dt)
# Umbral de |dPhi/dt| para enviar influencia
DPHI_DT_INFLUENCE_THRESHOLD = float(
    os.environ.get("MW_DPHI_DT_THRESHOLD", 1.0))


# --- Clases PhosWave y Electron ---
class PhosWave:
    """Representa el mecanismo de acoplamiento entre celdas (osciladores)."""

    # coef_transmision ahora es coef_acoplamiento
    def __init__(self, coef_acoplamiento=BASE_COUPLING_T):
        self.C = max(0.0, coef_acoplamiento)

    def ajustar_coeficientes(self, nuevos_C):
        """Ajusta el coeficiente de acoplamiento."""
        self.C = max(0.0, nuevos_C)
        logger.debug(
            "PhosWave coeficiente de acoplamiento ajustado a C=%.3f", self.C
        )


class Electron:
    """
    Representa el mecanismo de amortiguación local en cada celda (oscilador).
    """

    # coef_interaccion ahora es coef_amortiguacion
    def __init__(self, coef_amortiguacion=BASE_DAMPING_E):
        self.D = max(0.0, coef_amortiguacion)  # Coeficiente de amortiguación

    def ajustar_coeficientes(self, nuevos_D):
        """Ajusta el coeficiente de amortiguación."""
        self.D = max(0.0, nuevos_D)
        logger.debug(
            "Electron coeficiente de amortiguación ajustado a D=%.3f", self.D
        )


def apply_external_field_to_mesh(
    mesh_instance: HexCylindricalMesh,
    field_vector_map: List[List[List[float]]]
):
    """
    Aplica un campo vectorial externo (ej. campo V de ECU) a las celdas de la
    instancia de malla proporcionada, actualizando su q_vector usando
    interpolación bilineal.

    Args:
        mesh_instance (HexCylindricalMesh): La instancia de la malla a la que
            se aplicará el campo.
        field_vector_map (List[List[List[float]]]): Datos del campo vectorial.
    """
    if not mesh_instance or not mesh_instance.cells:
        logger.warning(
            "Intento de aplicar campo externo a malla no inicializada o vacía."
        )
        return

    # Convertir la lista de listas a un array NumPy para facilitar acceso
    try:
        field_vector_np = np.array(field_vector_map, dtype=float)
        if field_vector_np.ndim != 4 or field_vector_np.shape[-1] != 2:
            logger.error(
                "El campo vectorial externo no tiene el shape esperado "
                "[capas, filas, columnas, 2]. Recibido: %s",
                field_vector_np.shape
            )
            return
    except (ValueError, TypeError) as err:
        logger.error(
            "Error al convertir field_vector_map a NumPy array: %s", err
        )
        return

    logger.debug(
        "Aplicando campo vectorial externo (shape=%s) a la malla cilíndrica "
        "usando interpolación bilineal...",
        field_vector_np.shape)
    num_capas_torus, num_rows_torus, num_cols_torus, _ = \
        field_vector_np.shape

    if num_capas_torus <= 0 or num_rows_torus <= 0 or num_cols_torus <= 0:
        logger.error(
            "El campo vectorial externo tiene dimensiones inválidas (<= 0)."
        )
        return

    # La instancia de malla (mesh_instance)
    # no debe almacenar el campo completo.
    # Si necesitas este campo para otros cálculos
    # DENTRO de malla_watcher.py,
    # guárdalo en una variable global o pásalo como argumento.

    # Usar los métodos/atributos de la instancia de malla proporcionada
    all_mesh_cells = mesh_instance.get_all_cells()
    if not all_mesh_cells:
        logger.warning(
            "No hay celdas para aplicar campo externo."
        )
        return

    # Usar atributos min_z y max_z de la instancia de malla
    cylinder_z_min = mesh_instance.min_z
    cylinder_z_max = mesh_instance.max_z
    cylinder_height = cylinder_z_max - cylinder_z_min

    # Manejo de caso donde la altura del cilindro es cero o muy pequeña
    if cylinder_height <= EPSILON:
        if num_rows_torus > 1:
            logger.warning(
                "Altura del cilindro es cero o muy pequeña, la normalización "
                "Z para mapeo a filas del toroide puede no ser efectiva o "
                "dar resultados inesperados.")
        # Si la altura es cero, todas las celdas se mapearán a la misma fila

    update_count = 0
    for cell in all_mesh_cells:
        try:
            # Mapear coordenadas cilíndricas (theta, z) a coordenadas 2D
            col_float = (cell.theta / (2 * math.pi)) * num_cols_torus
            col_float = max(0.0, min(col_float, num_cols_torus - EPSILON))

            row_float = 0.0  # Default para malla plana
            if cylinder_height > EPSILON and num_rows_torus > 1:
                normalized_z = (cell.z - cylinder_z_min) / cylinder_height
                row_float = normalized_z * (num_rows_torus - 1)
                row_float = max(
                    0.0, min(row_float, num_rows_torus - 1.0 - EPSILON))
            elif num_rows_torus == 1:
                row_float = 0.0

            capa_idx_torus = 0

            r1 = math.floor(row_float)
            c1 = math.floor(col_float)

            # Asegurar que los índices estén dentro de los límites
            c1_idx = int(c1 % num_cols_torus)
            c2_idx = int((c1 + 1) % num_cols_torus)
            r1_idx = int(max(0, min(r1, num_rows_torus - 1)))
            r2_idx = int(max(0, min(r1 + 1, num_rows_torus - 1)))

            v11 = field_vector_np[capa_idx_torus, r1_idx, c1_idx, :]
            v12 = field_vector_np[capa_idx_torus, r1_idx, c2_idx, :]
            v21 = field_vector_np[capa_idx_torus, r2_idx, c1_idx, :]
            v22 = field_vector_np[capa_idx_torus, r2_idx, c2_idx, :]

            dr = row_float - r1
            dc = col_float - c1

            interp_vx = (
                v11[0] * (1 - dr) * (1 - dc) +
                v21[0] * dr * (1 - dc) +
                v12[0] * (1 - dr) * dc +
                v22[0] * dr * dc
            )

            interp_vy = (
                v11[1] * (1 - dr) * (1 - dc) +
                v21[1] * dr * (1 - dc) +
                v12[1] * (1 - dr) * dc +
                v22[1] * dr * dc
            )

            cell.q_vector = np.array([interp_vx, interp_vy], dtype=float)
            update_count += 1

        except IndexError:
            logger.warning(
                "Índice fuera de rango al acceder a field_vector_np. Celda "
                "(%d,%d) mapeada a (row_f=%.2f,col_f=%.2f) en capa %d. "
                "Índices calculados: r1_idx=%d, r2_idx=%d, c1_idx=%d, "
                "c2_idx=%d. Shape: %s",
                cell.q_axial, cell.r_axial, row_float, col_float,
                capa_idx_torus, r1_idx, r2_idx, c1_idx, c2_idx,
                field_vector_np.shape)
        except Exception:  # noqa: E722
            logger.error(
                "Error inesperado durante interpolación vectorial para celda "
                "(%d,%d)", cell.q_axial, cell.r_axial, exc_info=True)

    logger.debug(
        "Campo vectorial externo aplicado a %d/%d"
        "celdas mediante interpolación.",
        update_count, len(mesh_instance.cells)
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
        circumference_segments_target=int(
            os.environ.get("MW_CIRCUM_SEG", 6)),
        hex_size=float(os.environ.get("MW_HEX_SIZE", 1.0)),
        periodic_z=os.environ.get("MW_PERIODIC_Z", "True").lower() == "true")
    # Inicializar el flujo previo
    malla_cilindrica_global.previous_flux = 0.0
    logger.info(
        "Instancia global inicial (import time) creada con %d celdas.",
        len(malla_cilindrica_global.cells))
except Exception as err:
    logger.exception(
        "Error al inicializar HexCylindricalMesh global (import time). "
        "Será re-inicializada en __main__.", exc_info=err)
    malla_cilindrica_global = None


# --- Instancias de PhosWave y Electron ---
# Inicializadas con valores base.
resonador_global = PhosWave(
    coef_acoplamiento=float(os.environ.get("MW_BASE_T", 0.6))
)
electron_global = Electron(
    coef_amortiguacion=float(os.environ.get("MW_BASE_E", 0.1))
)


# --- Estado Agregado y Control ---
aggregate_state_lock = threading.Lock()
aggregate_state: Dict[str, Any] = {
    "avg_amplitude": 0.0,
    "max_amplitude": 0.0,
    "avg_velocity": 0.0,
    "max_velocity": 0.0,
    "avg_kinetic_energy": 0.0,
    "max_kinetic_energy": 0.0,
    "avg_activity_magnitude": 0.0,
    "max_activity_magnitude": 0.0,
    "cells_over_threshold": 0
}
control_lock = threading.Lock()
control_params: Dict[str, float] = {
    "phoswave_C": resonador_global.C,
    "electron_D": electron_global.D
}


# --- Lógica de Simulación (Propagación y Estabilización) ---
def simular_paso_malla():
    """Simula un paso de la dinámica de osciladores acoplados en la malla."""
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        logger.error(
            "Intento de simular paso en malla no inicializada o vacía.")
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


def update_aggregate_state():
    """Calcula y actualiza el estado agregado de la malla."""
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        with aggregate_state_lock:
            aggregate_state["avg_amplitude"] = 0.0
            aggregate_state["max_amplitude"] = 0.0
            aggregate_state["avg_velocity"] = 0.0
            aggregate_state["max_velocity"] = 0.0
            aggregate_state["avg_kinetic_energy"] = 0.0
            aggregate_state["max_kinetic_energy"] = 0.0
            aggregate_state["avg_activity_magnitude"] = 0.0
            aggregate_state["max_activity_magnitude"] = 0.0
            aggregate_state["cells_over_threshold"] = 0
        logger.debug(
            "Malla no inicializada o vacía, estado agregado reseteado.")
        return

    all_cells = mesh.get_all_cells()
    if not all_cells:
        with aggregate_state_lock:
            aggregate_state["avg_amplitude"] = 0.0
            aggregate_state["max_amplitude"] = 0.0
            aggregate_state["avg_velocity"] = 0.0
            aggregate_state["max_velocity"] = 0.0
            aggregate_state["avg_kinetic_energy"] = 0.0
            aggregate_state["max_kinetic_energy"] = 0.0
            aggregate_state["avg_activity_magnitude"] = 0.0
            aggregate_state["max_activity_magnitude"] = 0.0
            aggregate_state["cells_over_threshold"] = 0
        logger.debug("Lista de celdas vacía, estado agregado reseteado.")
        return

    # Calcular métricas individuales
    amplitudes = [cell.amplitude for cell in all_cells]
    velocities = [cell.velocity for cell in all_cells]
    # KE = 0.5 * m * v^2, m=1
    kinetic_energies = [0.5 * v**2 for v in velocities]
    activity_magnitudes = [
        math.sqrt(cell.amplitude**2 + cell.velocity**2)
        for cell in all_cells
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
        1 for mag in activity_magnitudes
        if mag > AMPLITUDE_INFLUENCE_THRESHOLD
    )

    with aggregate_state_lock:
        aggregate_state["avg_amplitude"] = avg_amp
        aggregate_state["max_amplitude"] = max_amp
        aggregate_state["avg_velocity"] = avg_vel
        aggregate_state["max_velocity"] = max_vel
        aggregate_state["avg_kinetic_energy"] = avg_ke
        aggregate_state["max_kinetic_energy"] = max_ke
        aggregate_state["avg_activity_magnitude"] = avg_activity
        aggregate_state["max_activity_magnitude"] = max_activity
        aggregate_state["cells_over_threshold"] = over_thresh

    logger.debug(
        "Estado agregado actualizado: AvgAmp=%.3f, MaxAmp=%.3f, AvgVel=%.3f, "
        "MaxVel=%.3f, AvgKE=%.3f, MaxKE=%.3f, AvgActivity=%.3f, "
        "MaxActivity=%.3f, OverThresh=%d", avg_amp, max_amp, avg_vel, max_vel,
        avg_ke, max_ke, avg_activity, max_activity, over_thresh)


def calculate_flux(mesh: HexCylindricalMesh) -> float:
    """
    Calcula una representación simplificada del flujo magnético a través de
    la malla. Suma una componente del campo vectorial externo (q_vector)
    sobre todas las celdas.
    """
    if not mesh or not mesh.cells:
        return 0.0

    flux_component_index = 1  # Índice para la componente 'vy' del vector
    total_flux = 0.0
    for cell in mesh.cells.values():
        if cell.q_vector is not None and cell.q_vector.shape == (2,):
            total_flux += cell.q_vector[flux_component_index]
    logger.debug(
        "Flujo calculado (suma de componente %d): %.3f",
        flux_component_index, total_flux)
    return total_flux


def fetch_and_apply_torus_field():
    """
    Obtiene el campo vectorial completo de matriz_ecu y lo aplica a la malla.
    """
    ecu_vector_field_url = f"{MATRIZ_ECU_BASE_URL}/api/ecu/field_vector"
    logger.debug(
        "Intentando obtener campo vectorial de ECU en %s",
        ecu_vector_field_url)
    try:
        response = requests.get(ecu_vector_field_url, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
        response_data_dict = response.json()

        if (
            isinstance(response_data_dict, dict) and
            response_data_dict.get("status") == "success" and
            "field_vector" in response_data_dict
        ):
            actual_field_vector_list = response_data_dict["field_vector"]
            mesh = malla_cilindrica_global
            if mesh:
                if mesh.cells:
                    logger.debug(
                        "Aplicando campo vectorial ECU (primer elemento: %s)",
                        actual_field_vector_list[0] if actual_field_vector_list else 'vacío')
                    apply_external_field_to_mesh(
                        mesh, actual_field_vector_list)
                    logger.info(
                        "Campo vectorial toroidal (V) obtenido y aplicado "
                        "exitosamente.")
                else:
                    logger.warning(
                        "Malla global no tiene celdas. No se aplicará el "
                        "campo vectorial obtenido.")
            else:
                logger.warning(
                    "Malla global no inicializada. No se pudo aplicar el "
                    "campo vectorial obtenido.")
        else:
            logger.error(
                "Respuesta JSON inválida, no exitosa, o 'field_vector' "
                "ausente de %s. Respuesta: %s", ecu_vector_field_url,
                response_data_dict)

    except requests.exceptions.Timeout:
        logger.error(
            "Timeout al intentar obtener campo vectorial de %s",
            ecu_vector_field_url)
    except requests.exceptions.RequestException as e:
        logger.error(
            "Error de red o HTTP al obtener campo vectorial de %s: %s",
            ecu_vector_field_url, e)
    except (ValueError, TypeError, json.JSONDecodeError) as err:
        logger.error(
            "Error al procesar/decodificar respuesta JSON de %s: %s",
            ecu_vector_field_url, err)
    except Exception:  # noqa: E722
        logger.exception(
            "Error inesperado al obtener/aplicar campo vectorial de %s",
            ecu_vector_field_url)


def map_cylinder_to_torus_coords(cell: Cell) -> Optional[Tuple[int, int, int]]:
    """
    Mapea coordenadas y estado de una celda del cilindro a coordenadas
    del toroide. Retorna None si el mapeo no es posible o dimensiones
    inválidas.
    """
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        logger.warning(
            "Malla no inicializada o vacía. Saltando mapeo a toroide.")
        return None

    if TORUS_NUM_CAPAS <= 0 or TORUS_NUM_FILAS <= 0 or TORUS_NUM_COLUMNAS <= 0:
        logger.error("Dimensiones del toroide inválidas para mapeo.")
        return None

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
        return None

    # Usar la MAGNITUD de la actividad para mapear a la capa
    activity_magnitude = math.sqrt(cell.amplitude**2 + cell.velocity**2)
    normalized_activity = min(
        1.0, max(0.0,
                 activity_magnitude / MAX_AMPLITUDE_FOR_NORMALIZATION))

    # Mayor actividad -> menor índice de capa (capas "más internas")
    capa = int(round((1.0 - normalized_activity) * (TORUS_NUM_CAPAS - 1)))
    capa = max(0, min(capa, TORUS_NUM_CAPAS - 1))

    return capa, row, col


def send_influence_to_torus(dphi_dt: float):
    """
    Envía una influencia a matriz_ecu basada en la tasa de cambio del flujo
    magnético (dPhi/dt).
    """
    # Definir una ubicación fija o representativa en el toroide
    target_capa = 0  # Capa más interna/crítica
    target_row = TORUS_NUM_FILAS // 2  # Fila central
    target_col = TORUS_NUM_COLUMNAS // 2  # Columna central

    influence_vector = [dphi_dt, 0.0]
    watcher_name = f"malla_watcher_dPhiDt{dphi_dt:.3f}"

    payload = {
        "capa": target_capa,
        "row": target_row,
        "col": target_col,
        "vector": influence_vector,
        "nombre_watcher": watcher_name
    }

    ecu_influence_url = f"{MATRIZ_ECU_BASE_URL}/api/ecu/influence"
    try:
        response = requests.post(
            ecu_influence_url, json=payload, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
        logger.info(
            "Influencia (dPhi/dt) enviada a %s en (%d, %d, %d). Payload: %s. "
            "Respuesta: %d", ecu_influence_url, target_capa, target_row,
            target_col, payload, response.status_code)
    except requests.exceptions.Timeout:
        logger.error(
            "Timeout al enviar influencia (dPhi/dt) a %s",
            ecu_influence_url)
    except requests.exceptions.RequestException as e:
        logger.error(
            "Error de red al enviar influencia (dPhi/dt) a %s: %s",
            ecu_influence_url, e)
    except Exception:  # noqa: E722
        logger.exception(
            "Error inesperado al enviar influencia (dPhi/dt) a %s",
            ecu_influence_url)


# --- Bucle de Simulación (Thread) ---
simulation_thread = None
stop_simulation_event = threading.Event()


def simulation_loop():
    logger.info("Iniciando bucle de simulación de malla...")
    step_count = 0
    dt = SIMULATION_INTERVAL

    mesh = malla_cilindrica_global
    if mesh is None:
        logger.error(
            "Malla global no inicializada. "
            "No se puede iniciar el bucle de simulación.")
        return

    if not hasattr(mesh, 'previous_flux'):
        mesh.previous_flux = 0.0
        logger.warning(
            "previous_flux no inicializado en malla_cilindrica_global. "
            "Inicializando a 0.0")

    while not stop_simulation_event.is_set():
        start_time = time.monotonic()
        step_count += 1
        logger.debug("--- Iniciando paso de simulación %d ---", step_count)

        try:
            logger.debug(
                "Paso %d: Obteniendo campo vectorial del toroide...",
                step_count)
            fetch_and_apply_torus_field()

            logger.debug(
                "Paso %d: Calculando flujo magnético...", step_count)
            current_flux = calculate_flux(mesh)

            dphi_dt = (current_flux - mesh.previous_flux) / \
                dt if dt > 0 else 0.0
            mesh.previous_flux = current_flux

            logger.debug(
                "Paso %d: Flujo actual=%.3f, dPhi/dt=%.3f",
                step_count, current_flux, dphi_dt)

            logger.debug(
                "Paso %d: Simulando dinámica interna de la malla...",
                step_count)
            simular_paso_malla()

            logger.debug(
                "Paso %d: Actualizando estado agregado...", step_count)
            update_aggregate_state()

            if abs(dphi_dt) > DPHI_DT_INFLUENCE_THRESHOLD:
                logger.info(
                    "Paso %d: |dPhi/dt|=%.3f supera umbral %.3f. Enviando "
                    "influencia...", step_count, abs(dphi_dt),
                    DPHI_DT_INFLUENCE_THRESHOLD)
                send_influence_to_torus(dphi_dt)
            else:
                logger.debug(
                    "Paso %d: |dPhi/dt|=%.3f no supera umbral para influenciar"
                    "toroide.", step_count, abs(dphi_dt))

        except Exception:  # noqa: E722
            logger.exception(
                "Error durante el paso de simulación %d de malla.", step_count)

        elapsed_time = time.monotonic() - start_time
        sleep_time = max(0, dt - elapsed_time)
        logger.debug(
            "--- Paso %d completado en %.3fs. Durmiendo por %.3fs ---",
            step_count, elapsed_time, sleep_time)
        if sleep_time > 0:
            stop_simulation_event.wait(sleep_time)

    logger.info("Bucle de simulación de malla detenido.")


AGENT_AI_REGISTER_URL = os.environ.get(
    "AGENT_AI_REGISTER_URL", "http://agent_ai:9000/api/register")
MAX_REGISTRATION_RETRIES = 5
RETRY_DELAY = 5


def register_with_agent_ai(
    module_name: str,
    module_url: str,
    health_url: str,
    module_type: str,
    aporta_a: str,
    naturaleza: str,
    description: str = ""
):
    """Intenta registrar este módulo con AgentAI, con reintentos."""
    payload = {
        "nombre": module_name,
        "url": module_url,
        "url_salud": health_url,
        "tipo": module_type,
        "aporta_a": aporta_a,
        "naturaleza_auxiliar": naturaleza,
        "descripcion": description
    }
    logger.info(
        "Intentando registrar '%s' en AgentAI (%s)...",
        module_name, AGENT_AI_REGISTER_URL)
    for attempt in range(MAX_REGISTRATION_RETRIES):
        try:
            response = requests.post(
                AGENT_AI_REGISTER_URL, json=payload, timeout=4.0)
            response.raise_for_status()
            if response.status_code == 200:
                logger.info(
                    "Registro de '%s' exitoso en AgentAI.", module_name)
                return True
            else:
                logger.warning(
                    "Registro de '%s' recibido con status %d. Respuesta: %s",
                    module_name, response.status_code, response.text)
        except requests.exceptions.RequestException as e:
            logger.error(
                "Error de conexión al intentar registrar '%s' "
                "(intento %d/%d): %s",
                module_name, attempt + 1, MAX_REGISTRATION_RETRIES, e)
        except Exception as err_info:
            logger.error(
                "Error inesperado durante el registro de '%s' "
                "(intento %d/%d): %s",
                module_name, attempt + 1, MAX_REGISTRATION_RETRIES, err_info)

        if attempt < MAX_REGISTRATION_RETRIES - 1:
            logger.info("Reintentando registro en %d segundos...", RETRY_DELAY)
            time.sleep(RETRY_DELAY)
        else:
            logger.error(
                "No se pudo registrar '%s' en AgentAI después de %d intentos.",
                module_name, MAX_REGISTRATION_RETRIES)
            return False
    return False


# --- Servidor Flask ---
app = Flask(__name__)


# --- Endpoints de Flask ---
@app.route('/api/health', methods=['GET'])
def health_check():
    sim_alive = simulation_thread.is_alive() if simulation_thread else False
    mesh = malla_cilindrica_global
    num_cells = len(mesh.cells) if mesh and mesh.cells else 0
    status = "success"
    message = "Malla_watcher operativo."

    connectivity_status = "N/A"
    min_neighbors = -1
    max_neighbors = -1

    if mesh is None:
        status = "error"
        message = "Error: Objeto HexCylindricalMesh global no inicializado."
    elif num_cells == 0:
        status = "error"
        message = "Error: Malla inicializada pero contiene 0 celdas."
    elif not sim_alive:
        status = "error"
        message = "Error: Hilo de simulación del resonador inactivo."
    else:
        try:
            connectivity_counts = mesh.verify_connectivity()
            if connectivity_counts:
                min_neighbors = min(connectivity_counts.keys())
                max_neighbors = max(connectivity_counts.keys())
                if max_neighbors > 6 or \
                   (min_neighbors < 3 and num_cells > 1):
                    connectivity_status = "error"
                    status = "error"
                    message = ("Error estructural: Problemas graves de "
                               "conectividad.")
                elif min_neighbors < 6 and num_cells > 1:
                    connectivity_status = "warning"
                    if status == "success":
                        status = "warning"
                        message = ("Advertencia: Posibles problemas de "
                                   "conectividad.")
                else:
                    connectivity_status = "ok"
            else:
                connectivity_status = "warning"
                if status == "success":
                    status = "warning"
                    message = "Advertencia: verify_connectivity retornó vacío."
        except Exception as err_info:
            logger.error("Error durante verify_connectivity: %s", err_info)
            connectivity_status = "error"
            status = "error"
            message = f"Error interno al verificar conectividad: {err_info}"

    # --- Construir Respuesta JSON Detallada ---
    response_data = {
        "status": status,
        "module": "Malla_watcher",
        "message": message,
        "details": {
            "mesh": {
                "initialized": mesh is not None,
                "num_cells": num_cells,
                "connectivity_status": connectivity_status,
                "min_neighbors": min_neighbors,
                "max_neighbors": max_neighbors,
                "z_periodic": mesh.periodic_z if mesh else None},
            "resonator_simulation": {
                "running": sim_alive,
            }
        }
    }

    http_status_code = 200
    if status == "warning":
        http_status_code = 503
    elif status == "error":
        http_status_code = 500

    logger.debug(
        "Health check response: Status=%s, HTTP=%d", status, http_status_code)
    return jsonify(response_data), http_status_code


@app.route('/api/state', methods=['GET'])
def get_malla_state():
    """
    Devuelve el estado agregado actual de la malla y parámetros de control.
    """
    state_data = {}
    with aggregate_state_lock:
        state_data.update(aggregate_state)

    with control_lock:
        state_data["control_params"] = {
            "phoswave_C": resonador_global.C,
            "electron_D": electron_global.D
        }

    mesh = malla_cilindrica_global
    state_data["num_cells"] = len(mesh.cells) if mesh and mesh.cells else 0

    logger.debug("Devolviendo estado agregado: %s", state_data)
    return jsonify({"status": "success", "state": state_data})


@app.route('/api/control', methods=['POST'])
def set_malla_control():
    data = request.get_json(silent=True)
    if data is None or "control_signal" not in data:
        logger.error("Solicitud a /api/control sin payload JSON válido o "
                     "'control_signal'")
        return jsonify({
            "status": "error",
            "message": "Payload JSON vacío, inválido o falta 'control_signal'"
        }), 400

    signal = data['control_signal']
    if not isinstance(signal, (int, float)):
        return jsonify({
            "status": "error",
            "message": "El campo 'control_signal' debe ser un número."
        }), 400

    with control_lock:
        new_C = max(0.0, BASE_COUPLING_T + K_GAIN_COUPLING * signal)
        new_D = max(0.0, BASE_DAMPING_E - K_GAIN_DAMPING * signal)
        resonador_global.ajustar_coeficientes(new_C)
        electron_global.ajustar_coeficientes(new_D)
        control_params["phoswave_C"] = resonador_global.C
        control_params["electron_D"] = electron_global.D

    logger.info(
        "Parámetros de control ajustados: C=%.3f, D=%.3f (señal=%.3f)",
        resonador_global.C, electron_global.D, signal)
    return jsonify({
        "status": "success",
        "message": "Parámetros ajustados",
        "current_params": control_params
    }), 200


@app.route('/api/malla', methods=['GET'])
def get_malla():
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        return jsonify({
            "status": "error",
            "message": "Malla no inicializada o vacía."
        }), 503

    return jsonify({
        "status": "success",
        "metadata": {
            "radius": mesh.radius,
            "num_cells": len(mesh.cells),
            "periodic_z": mesh.periodic_z,
            "z_bounds": {"min": mesh.min_z, "max": mesh.max_z}
        },
        "cells": [cell.to_dict() for cell in mesh.get_all_cells()]
    }), 200


@app.route("/api/malla/influence", methods=["POST"])
def aplicar_influencia_toroide_push():
    """
    Recibe el CAMPO VECTORIAL completo del toroide (matriz_ecu) y lo aplica
    a la malla cilíndrica actualizando los q_vector de las celdas usando
    interpolación bilineal. (Alternativa pasiva a
    fetch_and_apply_torus_field)

    Espera JSON: {"field_vector": List[List[List[float]]]}
    (Shape: [capas, filas, columnas, 2])
    """
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        logger.warning(
            "Malla no inicializada o vacía. Saltando aplicación de influencia."
        )
        return jsonify({
            "status": "warning",
            "message": "Malla no inicializada o vacía. Influencia no aplicada."
        }), 503

    logger.warning(
        "Recibida llamada a endpoint pasivo /api/malla/influence. "
        "Se recomienda usar el fetch activo.")
    data = request.get_json(silent=True)
    if data is None or "field_vector" not in data:
        logger.error(
            "Solicitud POST a /api/malla/influence sin 'field_vector'.")
        return jsonify({
            "status": "error",
            "message": "Payload JSON vacío, inválido o falta 'field_vector'"
        }), 400

    field_vector_data = data["field_vector"]

    try:
        apply_external_field_to_mesh(mesh, field_vector_data)
        logger.info(
            "Influencia del campo vectorial toroidal (push) aplicada"
            "correctamente.")
        return jsonify({
            "status": "success",
            "message": "CV externo aplicado a q_vector de las celdas."
        }), 200

    except ValueError as ve:
        logger.error(
            "Error de valor al procesar campo vectorial externo (push): %s",
            ve
        )
        return jsonify({
            "status": "error",
            "message": f"Error en los datos recibidos (push): {ve}"
        }), 400
    except Exception:
        logger.exception("Error inesperado al aplicar influencia del toroide.")
        return jsonify({
            "status": "error",
            "message": "Error interno al aplicar campo externo (push)."
        }), 500


@app.route('/api/event', methods=['POST'])
def receive_event():
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        logger.warning(
            "Malla no inicializada o vacía. Saltando aplicación de evento.")
        return jsonify({
            "status": "warning",
            "message": "Malla no inicializada o vacía. Evento no aplicado."
        }), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Payload JSON vacío o inválido"
        }), 400

    logger.info("Evento recibido: %s", data)
    if data.get("type") == "pulse" and "coords" in data and \
       "magnitude" in data:
        q_coord = data["coords"].get("q")
        r_coord = data["coords"].get("r")
        try:
            mag = float(data.get("magnitude", 1.0))
        except (ValueError, TypeError):
            return jsonify({
                "status": "error",
                "message": "Magnitud inválida"
            }), 400

        if q_coord is not None and r_coord is not None:
            cell = mesh.get_cell(q_coord, r_coord)
            if cell:
                logger.info(
                    "Celda (%d,%d) - Velocidad antes: %.3f",
                    q_coord, r_coord, cell.velocity)
                cell.velocity += mag
                logger.info(
                    "Celda (%d,%d) - Velocidad después: %.3f",
                    q_coord, r_coord, cell.velocity)
            else:
                logger.warning(
                    "No se encontró celda (%d,%d) para aplicar evento.",
                    q_coord, r_coord)
                return jsonify({
                    "status": "warning",
                    "message": f"Celda ({q_coord},{r_coord}) no encontrada."
                }), 404
        else:
            return jsonify({
                "status": "error",
                "message": "Coordenadas 'q' o 'r' faltantes o inválidas."
            }), 400
    else:
        return jsonify({
            "status": "error",
            "message": "Tipo de evento no soportado o datos incompletos."
        }), 400

    return jsonify({"status": "success", "message": "Evento procesado"}), 200


@app.route("/api/error", methods=["POST"])
def receive_error():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "status": "error",
            "message": "Payload JSON vacío o inválido"
        }), 400

    logger.error("Error reportado desde otro servicio: %s", data)
    return jsonify({"status": "success", "message": "Error procesado"}), 200


@app.route('/api/config', methods=['GET'])
def get_config():
    mesh = malla_cilindrica_global
    return jsonify({
        "status": "success",
        "config": {
            "malla_config": {
                "radius": float(os.environ.get("MW_RADIUS", 5.0)),
                "height_segments": int(os.environ.get("MW_HEIGHT_SEG", 6)),
                "circumference_segments_target":
                    int(os.environ.get("MW_CIRCUM_SEG", 12)),
                "circumference_segments_actual":
                    mesh.circumference_segments_actual if mesh else None,
                "hex_size": float(os.environ.get("MW_HEX_SIZE", 1.0)),
                "periodic_z":
                    os.environ.get("MW_PERIODIC_Z", "True").lower() == "true"},
            "communication_config": {
                "matriz_ecu_url": MATRIZ_ECU_BASE_URL,
                "torus_dims":
                    f"{TORUS_NUM_CAPAS}x{TORUS_NUM_FILAS}x{TORUS_NUM_COLUMNAS}",
                "influence_threshold": AMPLITUDE_INFLUENCE_THRESHOLD,
                "max_activity_normalization":
                    MAX_AMPLITUDE_FOR_NORMALIZATION},
            "simulation_config": {
                "interval": SIMULATION_INTERVAL,
                "dphi_dt_influence_threshold": DPHI_DT_INFLUENCE_THRESHOLD},
            "control_config": {
                "base_coupling_t": BASE_COUPLING_T,
                "base_damping_e": BASE_DAMPING_E,
                "k_gain_coupling": K_GAIN_COUPLING,
                "k_gain_damping": K_GAIN_DAMPING,
                "current_coupling_C": resonador_global.C,
                "current_damping_D": electron_global.D
            }
        }
    }), 200


# --- Punto de Entrada Principal ---
if __name__ == "__main__":
    # 1. Leer las variables de configuración REALES desde el entorno
    MESH_RADIUS_REAL = float(os.environ.get("MW_RADIUS", 5.0))
    MESH_HEIGHT_SEGMENTS_REAL = int(os.environ.get("MW_HEIGHT_SEG", 6))
    MESH_CIRCUMFERENCE_SEGMENTS_REAL = \
        int(os.environ.get("MW_CIRCUM_SEG", 12))
    MESH_HEX_SIZE_REAL = float(os.environ.get("MW_HEX_SIZE", 1.0))
    MESH_PERIODIC_Z_REAL = \
        os.environ.get("MW_PERIODIC_Z", "True").lower() == "true"

    MATRIZ_ECU_BASE_URL_REAL = \
        os.environ.get("MATRIZ_ECU_URL", "http://ecu:8000")
    TORUS_NUM_CAPAS_REAL = int(os.environ.get("TORUS_NUM_CAPAS", 3))
    TORUS_NUM_FILAS_REAL = int(os.environ.get("TORUS_NUM_FILAS", 4))
    TORUS_NUM_COLUMNAS_REAL = int(os.environ.get("TORUS_NUM_COLUMNAS", 5))
    AMPLITUDE_INFLUENCE_THRESHOLD_REAL = \
        float(os.environ.get("MW_INFLUENCE_THRESHOLD", 5.0))
    MAX_AMPLITUDE_FOR_NORMALIZATION_REAL = \
        float(os.environ.get("MW_MAX_AMPLITUDE_NORM", 20.0))
    REQUESTS_TIMEOUT_REAL = \
        float(os.environ.get("MW_REQUESTS_TIMEOUT", 2.0))
    BASE_COUPLING_T_REAL = float(os.environ.get("MW_BASE_T", 0.6))
    BASE_DAMPING_E_REAL = float(os.environ.get("MW_BASE_E", 0.1))
    K_GAIN_COUPLING_REAL = float(os.environ.get("MW_K_GAIN_T", 0.1))
    K_GAIN_DAMPING_REAL = float(os.environ.get("MW_K_GAIN_E", 0.05))
    SIMULATION_INTERVAL_REAL = \
        float(os.environ.get("MW_SIM_INTERVAL", 0.5))
    DPHI_DT_INFLUENCE_THRESHOLD_REAL = \
        float(os.environ.get("MW_DPHI_DT_THRESHOLD", 1.0))

    # 2. Loguear la configuración REAL
    logger.info(
        "Configuración Malla REAL: R=%.1f, HSeg=%d, CSeg=%d, Hex=%.1f, "
        "PerZ=%s", MESH_RADIUS_REAL, MESH_HEIGHT_SEGMENTS_REAL,
        MESH_CIRCUMFERENCE_SEGMENTS_REAL, MESH_HEX_SIZE_REAL,
        MESH_PERIODIC_Z_REAL)
    logger.info(
        "Configuración Comms REAL: ECU_URL=%s, Torus=%dx%dx%d, InfThr=%.1f",
        MATRIZ_ECU_BASE_URL_REAL, TORUS_NUM_CAPAS_REAL,
        TORUS_NUM_FILAS_REAL, TORUS_NUM_COLUMNAS_REAL,
        AMPLITUDE_INFLUENCE_THRESHOLD_REAL)
    logger.info(
        "Configuración Control REAL: BaseC=%.1f, BaseD=%.1f, GainC=%.1f, "
        "GainD=%.1f", BASE_COUPLING_T_REAL, BASE_DAMPING_E_REAL,
        K_GAIN_COUPLING_REAL, K_GAIN_DAMPING_REAL)
    logger.info(
        "Configuración Simulación REAL: Interval=%.1fs, dPhi/dt "
        "Threshold=%.1f", SIMULATION_INTERVAL_REAL,
        DPHI_DT_INFLUENCE_THRESHOLD_REAL)
    logger.info(
        "Configuración Normalización Influencia REAL: MaxActivityNorm=%.1f",
        MAX_AMPLITUDE_FOR_NORMALIZATION_REAL)

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
            "HexCylindricalMesh...")
        malla_cilindrica_global = HexCylindricalMesh(
            radius=MESH_RADIUS_REAL,
            height_segments=MESH_HEIGHT_SEGMENTS_REAL,
            circumference_segments_target=MESH_CIRCUMFERENCE_SEGMENTS_REAL,
            hex_size=MESH_HEX_SIZE_REAL,
            periodic_z=MESH_PERIODIC_Z_REAL)
        if not malla_cilindrica_global.cells:
            logger.error("¡La malla REAL se inicializó pero está vacía!")
            exit(1)
        else:
            logger.info(
                "Malla REAL inicializada con %d celdas.",
                len(malla_cilindrica_global.cells))
    except Exception:  # noqa: E722
        logger.exception(
            "Error crítico al RE-inicializar HexCylindricalMesh global REAL.")
        exit(1)

    resonador_global = PhosWave(coef_acoplamiento=BASE_COUPLING_T)
    electron_global = Electron(coef_amortiguacion=BASE_DAMPING_E)

    if not hasattr(malla_cilindrica_global, 'previous_flux'):
        malla_cilindrica_global.previous_flux = 0.0
        logger.info("previous_flux inicializado a 0.0 en instancia REAL.")
    logger.debug(
        "previous_flux en instancia REAL: %.1f",
        malla_cilindrica_global.previous_flux)

    with control_lock:
        control_params["phoswave_C"] = resonador_global.C
        control_params["electron_D"] = electron_global.D
    logger.info(
        "Parámetros de control iniciales REALES: C=%.3f, D=%.3f",
        control_params['phoswave_C'], control_params['electron_D'])

    # 4. Registro con AgentAI
    MODULE_NAME = "malla_watcher"
    SERVICE_PORT = int(os.environ.get("PORT", 5001))
    MODULE_URL = f"http://{MODULE_NAME}:{SERVICE_PORT}"
    HEALTH_URL = f"{MODULE_URL}/api/health"
    APORTA_A = "matriz_ecu"
    NATURALEZA = "modulador"
    DESCRIPTION = (
        "Simulador de malla hexagonal cilíndrica (osciladores acoplados) "
        "acoplado a ECU, influye basado en inducción electromagnética.")

    registration_successful = register_with_agent_ai(
        MODULE_NAME, MODULE_URL, HEALTH_URL, "auxiliar", APORTA_A,
        NATURALEZA, DESCRIPTION)
    if not registration_successful:
        logger.warning(
            "El módulo '%s' continuará sin registro exitoso en AgentAI.",
            MODULE_NAME)

    # 5. Iniciar Hilo de Simulación y Servidor Flask
    logger.info("Creando e iniciando hilo de simulación de malla...")
    stop_simulation_event.clear()
    simulation_thread = threading.Thread(
        target=simulation_loop, daemon=True, name="MallaSimLoop")
    simulation_thread.start()
    logger.info("Hilo de simulación iniciado.")

    logger.info("Iniciando servicio Flask de malla_watcher en puerto %d",
                SERVICE_PORT)
    app.run(host="0.0.0.0", port=SERVICE_PORT,
            debug=False, use_reloader=False)

    # Código de limpieza al detener el servidor
    logger.info(
        "Señal de detención recibida. Deteniendo hilo de simulación."
    )
    stop_simulation_event.set()
    if simulation_thread:
        simulation_thread.join(timeout=SIMULATION_INTERVAL * 3 + 5)
        if simulation_thread.is_alive():
            logger.warning("El hilo de simulación no terminó a tiempo.")
    logger.info("Servicio malla_watcher finalizado.")

# [end of watchers/watchers_tools/malla_watcher/malla_watcher.py]
