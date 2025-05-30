#!/usr/bin/env python3
"""
malla_watcher.py

Define una malla hexagonal cilíndrica inspirada en la estructura del grafeno, 
modelada como un sistema de osciladores acoplados. La malla interactúa con un 
campo vectorial externo (proveniente de ECU) que modula el acoplamiento entre 
osciladores, y genera influencias sobre dicho campo basadas en la tasa de cambio 
del flujo del campo a través de la malla (analogía de inducción electromagnética).

Componentes clave:
- **Cell**: Representa un oscilador en la malla con estado (amplitud, velocidad) 
  y campo externo local (q_vector).
- **HexCylindricalMesh** (importado desde `cilindro_grafenal`): Gestiona la malla 
  hexagonal cilíndrica con validación de conectividad y condiciones de contorno 
  periódicas.

Interacciones:
1. **ECU → Malla**: Obtiene periódicamente el campo vectorial de ECU y lo aplica 
   a las celdas mediante interpolación.
2. **Malla → ECU**: Envía influencias al toroide basadas en la tasa de cambio del 
   flujo del campo de ECU a través de la malla (dΦ/dt).

Dependencias:
- `cilindro_grafenal.HexCylindricalMesh`: Clase central para la generación y 
  validación de la estructura digital del cilindro.
"""

# import json as _json # Mantenido por si acaso, aunque ErrorResponse se elimina
import math
import logging
import requests
import time
import threading
import os
from flask import Flask, request, jsonify
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter
from os import environ
from enum import Enum
# Importar desde cilindro_grafenal
from .utils.cilindro_grafenal import HexCylindricalMesh, Cell

# --- Configuración del Logging ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("malla_watcher")
if not logger.hasHandlers():
    handler = logging.FileHandler(os.path.join(log_dir, "malla_watcher.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    # logger.setLevel(logging.INFO) # Mantener nivel INFO por defecto para ejecución normal
    # Los logs DEBUG en métodos como _initialize_mesh o get_neighbor_cells
    # solo se mostrarán si el nivel del logger se cambia a DEBUG (ej. en desarrollo)
# Pequeña tolerancia para comparaciones de flotantes
EPSILON = 1e-9

# --- Constantes de Configuración para Comunicación ---
MATRIZ_ECU_BASE_URL = os.environ.get("MATRIZ_ECU_URL", "http://ecu:8000")
TORUS_NUM_CAPAS = int(os.environ.get("TORUS_NUM_CAPAS", 3))
TORUS_NUM_FILAS = int(os.environ.get("TORUS_NUM_FILAS", 4))
TORUS_NUM_COLUMNAS = int(os.environ.get("TORUS_NUM_COLUMNAS", 5))
AMPLITUDE_INFLUENCE_THRESHOLD = float(os.environ.get("MW_INFLUENCE_THRESHOLD", 5.0)) # Umbral para métrica de actividad
MAX_AMPLITUDE_FOR_NORMALIZATION = float(os.environ.get("MW_MAX_AMPLITUDE_NORM", 20.0)) # Valor para normalizar métrica de actividad
REQUESTS_TIMEOUT = float(os.environ.get("MW_REQUESTS_TIMEOUT", 2.0))

# --- Constantes de Configuración para Control ---
BASE_COUPLING_T = float(os.environ.get("MW_BASE_T", 0.6))
K_GAIN_COUPLING = float(os.environ.get("MW_K_GAIN_T", 0.1))
BASE_DAMPING_E = float(os.environ.get("MW_BASE_E", 0.1))
K_GAIN_DAMPING = float(os.environ.get("MW_K_GAIN_E", 0.05))

# --- Constantes de Configuración para Simulación ---
SIMULATION_INTERVAL = float(os.environ.get("MW_SIM_INTERVAL", 0.5)) # Segundos (esto es dt)
DPHI_DT_INFLUENCE_THRESHOLD = float(os.environ.get("MW_DPHI_DT_THRESHOLD", 1.0)) # Umbral de |dPhi/dt| para enviar influencia

# --- Clases PhosWave y Electron ---
class PhosWave:
    """
    Representa el mecanismo de acoplamiento entre celdas (osciladores).
    """
    # coef_transmision ahora es coef_acoplamiento
    def __init__(self, coef_acoplamiento=BASE_COUPLING_T):
        self.C = max(0.0, coef_acoplamiento)

    def ajustar_coeficientes(self, nuevos_C):
        """Ajusta el coeficiente de acoplamiento."""
        self.C = max(0.0, nuevos_C)
        logger.debug(f"PhosWave coeficiente de acoplamiento ajustado a C={self.C:.3f}")

class Electron:
    """
    Representa el mecanismo de amortiguación local en cada celda (oscilador).
    """
    # coef_interaccion ahora es coef_amortiguacion
    def __init__(self, coef_amortiguacion=BASE_DAMPING_E):
        self.D = max(0.0, coef_amortiguacion) # Coeficiente de amortiguación

    def ajustar_coeficientes(self, nuevos_D):
        """Ajusta el coeficiente de amortiguación."""
        self.D = max(0.0, nuevos_D)
        logger.debug(f"Electron coeficiente de amortiguación ajustado a D={self.D:.3f}")

def apply_external_field_to_mesh(mesh_instance: HexCylindricalMesh, field_vector_map: List[List[List[float]]]):
    """
    Aplica un campo vectorial externo (ej. campo V de ECU) a las celdas de la instancia de malla proporcionada,
    actualizando su q_vector usando interpolación bilineal.

    Args:
        mesh_instance (HexCylindricalMesh): La instancia de la malla a la que se aplicará el campo.
        field_vector_map (List[List[List[float]]]): Datos del campo vectorial.
    """
    if not mesh_instance or not mesh_instance.cells: # Usar el argumento mesh_instance
        logger.warning("Intento de aplicar campo externo a malla no inicializada o vacía.")
        return # Añadir return para salir si la condición se cumple

    # Convertir la lista de listas a un array NumPy para facilitar el acceso e interpolación
    try:
        field_vector_np = np.array(field_vector_map, dtype=float)
        if field_vector_np.ndim != 4 or field_vector_np.shape[-1] != 2:
             logger.error(f"El campo vectorial externo no tiene el shape esperado [capas, filas, columnas, 2]. Recibido: {field_vector_np.shape}")
             return
    except (ValueError, TypeError) as e:
         logger.error(f"Error al convertir field_vector_map a NumPy array: {e}")
         return

    logger.debug(f"Aplicando campo vectorial externo (shape={field_vector_np.shape}) a la malla cilíndrica usando interpolación bilineal...")
    num_capas_torus, num_rows_torus, num_cols_torus, _ = field_vector_np.shape

    if num_capas_torus <= 0 or num_rows_torus <= 0 or num_cols_torus <= 0:
        logger.error("El campo vectorial externo tiene dimensiones inválidas (<= 0).")
        return

    # ELIMINAR: self.external_vector_field = field_vector_map
    # La instancia de malla (mesh_instance) no debe almacenar el campo completo.
    # Si necesitas este campo para otros cálculos DENTRO de malla_watcher.py,
    # guárdalo en una variable global de malla_watcher.py o pásalo como argumento.

    # Usar los métodos/atributos de la instancia de malla proporcionada
    all_mesh_cells = mesh_instance.get_all_cells()
    if not all_mesh_cells: # Chequeo adicional por si get_all_cells() devuelve vacío aunque mesh_instance.cells no lo sea
        logger.warning("No hay celdas para aplicar campo externo (obtenido de get_all_cells).")
        return
    
    # Usar los atributos min_z y max_z de la instancia de malla,
    # que son calculados por cilindro_grafenal.HexCylindricalMesh durante su inicialización.
    cylinder_z_min = mesh_instance.min_z
    cylinder_z_max = mesh_instance.max_z
    cylinder_height = cylinder_z_max - cylinder_z_min
    
    # Manejo de caso donde la altura del cilindro es cero o muy pequeña
    if cylinder_height <= EPSILON:
        if num_rows_torus > 1: # Solo advertir si el toroide tiene múltiples filas, implicando una dimensión Z
            logger.warning("Altura del cilindro es cero o muy pequeña, la normalización Z para mapeo a filas del toroide puede no ser efectiva o dar resultados inesperados.")
        # Si la altura es cero, todas las celdas se mapearán a la misma fila del toroide (row_float = 0.0).
        # Esto es manejado por la lógica de `normalized_z` y `row_float` más abajo.

    update_count = 0
    for cell in all_mesh_cells: # Iterar sobre las celdas de la instancia de malla
        try:
            # Mapear coordenadas cilíndricas (theta, z) a coordenadas 2D (col, row) del toroide
            col_float = (cell.theta / (2 * math.pi)) * num_cols_torus
            col_float = max(0.0, min(col_float, num_cols_torus - EPSILON)) # Clamp

            row_float = 0.0 # Default para malla plana o si no hay altura en el toroide
            if cylinder_height > EPSILON and num_rows_torus > 1: # Solo normalizar si hay altura y el toroide tiene filas
                normalized_z = (cell.z - cylinder_z_min) / cylinder_height
                row_float = normalized_z * (num_rows_torus - 1) # Mapear a [0, num_rows_torus - 1]
                row_float = max(0.0, min(row_float, num_rows_torus - 1.0 - EPSILON)) # Clamp
            elif num_rows_torus == 1: # Si el toroide solo tiene una fila, todas las celdas mapean a ella
                row_float = 0.0
            # Si num_rows_torus es 0, ya se retornó antes.

            capa_idx_torus = 0 

            r1 = math.floor(row_float)
            c1 = math.floor(col_float)

            # Asegurar que los índices estén dentro de los límites del array field_vector_np
            c1_idx = int(c1 % num_cols_torus) # c1 puede ser num_cols_torus si col_float es exactamente num_cols_torus
            c2_idx = int((c1 + 1) % num_cols_torus)
            r1_idx = int(max(0, min(r1, num_rows_torus - 1)))
            r2_idx = int(max(0, min(r1 + 1, num_rows_torus - 1)))


            v11 = field_vector_np[capa_idx_torus, r1_idx, c1_idx, :]
            v12 = field_vector_np[capa_idx_torus, r1_idx, c2_idx, :]
            v21 = field_vector_np[capa_idx_torus, r2_idx, c1_idx, :]
            v22 = field_vector_np[capa_idx_torus, r2_idx, c2_idx, :]

            dr = row_float - r1
            dc = col_float - c1

            interp_vx = (v11[0] * (1 - dr) * (1 - dc) +
                         v21[0] * dr * (1 - dc) +
                         v12[0] * (1 - dr) * dc +
                         v22[0] * dr * dc)

            interp_vy = (v11[1] * (1 - dr) * (1 - dc) +
                         v21[1] * dr * (1 - dc) +
                         v12[1] * (1 - dr) * dc +
                         v22[1] * dr * dc)

            cell.q_vector = np.array([interp_vx, interp_vy], dtype=float)
            update_count += 1

        except IndexError:
             logger.warning(f"Índice fuera de rango al acceder a field_vector_np. Celda ({cell.q_axial},{cell.r_axial}) mapeada a (row_f={row_float:.2f},col_f={col_float:.2f}) en capa {capa_idx_torus}. Índices calculados: r1_idx={r1_idx}, r2_idx={r2_idx}, c1_idx={c1_idx}, c2_idx={c2_idx}. Shape del campo: {field_vector_np.shape}")
        except Exception as e:
             logger.error(f"Error inesperado durante interpolación vectorial para celda ({cell.q_axial},{cell.r_axial}): {e}", exc_info=True)

    logger.debug(f"Campo vectorial externo aplicado a {update_count}/{len(mesh_instance.cells)} celdas mediante interpolación.") # Usar mesh_instance

# --- Instancia Global de la Malla Cilíndrica ---
MESH_RADIUS = float(os.environ.get("MW_RADIUS", 5.0))
MESH_HEIGHT_SEGMENTS = int(os.environ.get("MW_HEIGHT_SEG", 2))
MESH_CIRCUMFERENCE_SEGMENTS = int(os.environ.get("MW_CIRCUM_SEG", 6))
MESH_HEX_SIZE = float(os.environ.get("MW_HEX_SIZE", 1.0))
MESH_PERIODIC_Z = os.environ.get("MW_PERIODIC_Z", "True").lower() == "true"

# La inicialización global usará la nueva definición de Cell
try:
    malla_cilindrica_global = HexCylindricalMesh(
        radius=float(os.environ.get("MW_RADIUS", 5.0)), # Usar ENV defaults aquí también
        height_segments=int(os.environ.get("MW_HEIGHT_SEG", 3)), # Usar ENV defaults aquí también
        circumference_segments_target=int(os.environ.get("MW_CIRCUM_SEG", 6)), # Usar ENV defaults aquí también
        hex_size=float(os.environ.get("MW_HEX_SIZE", 1.0)), # Usar ENV defaults aquí también
        periodic_z=os.environ.get("MW_PERIODIC_Z", "True").lower() == "true" # Usar ENV defaults aquí también
    )
    # Inicializar el flujo previo
    malla_cilindrica_global.previous_flux = 0.0 # Inicializar el flujo previo a 0
    logger.info(f"Instancia global inicial (import time) creada con {len(malla_cilindrica_global.cells)} celdas.")
except Exception as e:
    logger.exception("Error al inicializar HexCylindricalMesh global (import time). Será re-inicializada en __main__.")
    # Si falla aquí, la instancia global podría ser None o parcial.
    # La lógica en __main__ y endpoints debe manejar esto.
    malla_cilindrica_global = None # Asegurar que es None si falla

# --- Instancias de PhosWave y Electron ---
# Inicializadas con valores base.
resonador_global = PhosWave(coef_acoplamiento=float(os.environ.get("MW_BASE_T", 0.6)))
electron_global = Electron(coef_amortiguacion=float(os.environ.get("MW_BASE_E", 0.1)))

# --- Estado Agregado y Control ---
aggregate_state_lock = threading.Lock()
aggregate_state: Dict[str, Any] = {
    "avg_amplitude": 0.0, # Mantener para referencia, pero la métrica clave será la magnitud combinada
    "max_amplitude": 0.0, # Mantener para referencia
    "avg_velocity": 0.0, # Añadir velocidad promedio
    "max_velocity": 0.0, # Añadir velocidad máxima
    "avg_kinetic_energy": 0.0, # Añadir energía cinética promedio
    "max_kinetic_energy": 0.0, # Añadir energía cinética máxima
    "avg_activity_magnitude": 0.0, # Añadir métrica combinada (sqrt(amp^2 + vel^2))
    "max_activity_magnitude": 0.0, # Añadir métrica combinada máxima
    "cells_over_threshold": 0 # Ahora basado en activity_magnitude
}
control_lock = threading.Lock()
# control_params ahora reflejará C y D
control_params: Dict[str, float] = {
    "phoswave_C": resonador_global.C, # Renombrado
    "electron_D": electron_global.D   # Renombrado
}

# --- Lógica de Simulación (Propagación y Estabilización) ---
def simular_paso_malla():
    """Simula un paso de la dinámica de osciladores acoplados en la malla."""
    # Acceder directamente a la malla global (que puede ser None o un mock en tests)
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
         logger.error("Intento de simular paso en malla no inicializada o vacía.")
         return

    dt = SIMULATION_INTERVAL # Paso de tiempo
    all_cells = mesh.get_all_cells()
    num_cells = len(all_cells)
    if num_cells == 0: return

    next_amplitudes = [0.0] * num_cells
    next_velocities = [0.0] * num_cells

    # Asumimos masa = 1 para simplificar: Force = Acceleration

    # Fase 1: Calcular fuerzas y nuevas velocidades para todas las celdas
    for i, cell_i in enumerate(all_cells):
        # 1a: Fuerza de Acoplamiento (de vecinos)
        coupling_force = 0.0
        # Llamar al método get_neighbor_cells en la instancia global
        neighbors = mesh.get_neighbor_cells(cell_i.q_axial, cell_i.r_axial)

        # Coeficiente de acoplamiento modulado por la MAGNITUD del campo vectorial local (q_vector)
        # Aseguramos que el modulador no sea negativo (ej: 1 + |q_v|)
        # Opcional: modular por una componente específica si tiene más sentido físico
        modulator = 1.0 + np.linalg.norm(cell_i.q_vector) # Usar la norma del vector
        # Acceder directamente a resonador_global
        modulated_coupling_C = resonador_global.C * max(0.0, modulator)

        for cell_j in neighbors:
            # Fuerza de j sobre i es proporcional a (pos_j - pos_i)
            coupling_force += modulated_coupling_C * (cell_j.amplitude - cell_i.amplitude)

        # 1b: Fuerza de Amortiguación (local)
        # Acceder directamente a electron_global
        damping_force = - electron_global.D * cell_i.velocity

        # 1c: Fuerza Neta
        net_force = coupling_force + damping_force
        # Nota: No hay fuerza externa constante o "restoring force" K*x por ahora,
        # la dinámica emerge del acoplamiento y la amortiguación.
        # El q_vector modula el acoplamiento, no es una fuerza directa.

        # 1d: Calcular nueva velocidad (Euler)
        acceleration = net_force # masa = 1
        next_velocities[i] = cell_i.velocity + acceleration * dt

    # Fase 2: Calcular nuevas amplitudes usando las nuevas velocidades
    for i, cell_i in enumerate(all_cells):
        next_amplitudes[i] = cell_i.amplitude + next_velocities[i] * dt

    # Fase 3: Actualizar los estados de las celdas
    for i, cell_i in enumerate(all_cells):
        cell_i.amplitude = next_amplitudes[i]
        cell_i.velocity = next_velocities[i]

    logger.debug(f"Paso de simulación de osciladores acoplados completado para {num_cells} celdas.")

def update_aggregate_state():
    """Calcula y actualiza el estado agregado de la malla, incluyendo métricas de energía/actividad."""
    global aggregate_state
    # Acceder directamente a la malla global (que puede ser None o un mock en tests)
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
        # Si no hay malla o está vacía, resetear el estado agregado
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
        logger.debug("Malla no inicializada o vacía, estado agregado reseteado.")
        return

    all_cells = mesh.get_all_cells()
    if not all_cells:
         # Si la lista de celdas está vacía (aunque self.cells no lo esté, ej. por un mock raro)
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
    kinetic_energies = [0.5 * v**2 for v in velocities] # KE = 0.5 * m * v^2, m=1
    # Métrica de "actividad total" combinada: sqrt(amplitude^2 + velocity^2)
    activity_magnitudes = [math.sqrt(cell.amplitude**2 + cell.velocity**2) for cell in all_cells]

    num_cells = len(all_cells)
    avg_amp = sum(amplitudes) / num_cells
    max_amp = max(amplitudes, default=0.0) # Usar default para listas vacías
    avg_vel = sum(velocities) / num_cells
    max_vel = max(velocities, default=0.0)
    avg_ke = sum(kinetic_energies) / num_cells
    max_ke = max(kinetic_energies, default=0.0)
    avg_activity = sum(activity_magnitudes) / num_cells
    max_activity = max(activity_magnitudes, default=0.0)

    # El umbral de influencia ahora se compara con la métrica de actividad total
    over_thresh = sum(1 for mag in activity_magnitudes if mag > AMPLITUDE_INFLUENCE_THRESHOLD)

    with aggregate_state_lock:
        # Actualizar el estado agregado
        aggregate_state["avg_amplitude"] = avg_amp
        aggregate_state["max_amplitude"] = max_amp
        aggregate_state["avg_velocity"] = avg_vel
        aggregate_state["max_velocity"] = max_vel
        aggregate_state["avg_kinetic_energy"] = avg_ke
        aggregate_state["max_kinetic_energy"] = max_ke
        aggregate_state["avg_activity_magnitude"] = avg_activity
        aggregate_state["max_activity_magnitude"] = max_activity
        aggregate_state["cells_over_threshold"] = over_thresh

    logger.debug(f"Estado agregado actualizado: AvgAmp={avg_amp:.3f}, MaxAmp={max_amp:.3f}, AvgVel={avg_vel:.3f}, MaxVel={max_vel:.3f}, AvgKE={avg_ke:.3f}, MaxKE={max_ke:.3f}, AvgActivity={avg_activity:.3f}, MaxActivity={max_activity:.3f}, OverThresh={over_thresh}")

# Método para calcular el flujo magnético a través de la malla
def calculate_flux(mesh: HexCylindricalMesh) -> float:
    """
    Calcula una representación simplificada del flujo magnético a través de la malla.
    Suma una componente del campo vectorial externo (q_vector) sobre todas las celdas.
    """
    if not mesh or not mesh.cells:
        return 0.0

    # Elegir una componente del vector para representar el flujo "a través" del cilindro.
    # Si q_vector = [vx, vy], donde vx es toroidal y vy es poloidal/vertical,
    # el flujo a través de una sección transversal perpendicular al eje Z (eje del cilindro)
    # podría estar relacionado con la suma de la componente vy.
    # O el flujo a través de la superficie cilíndrica podría estar relacionado con la componente vx.
    # Usemos la componente vy (índice 1) como una representación simple del flujo "vertical" o "poloidal"
    # que atraviesa la malla.
    flux_component_index = 1 # Índice para la componente 'vy' del vector

    total_flux = 0.0
    for cell in mesh.cells.values():
        # Sumar la componente elegida del vector de campo externo en cada celda
        # Podríamos ponderar por el "área" representada por la celda, pero asumimos área unitaria por ahora.
        if cell.q_vector is not None and cell.q_vector.shape == (2,):
             total_flux += cell.q_vector[flux_component_index]
        # else: logger.warning(f"Celda ({cell.q_axial},{cell.r_axial}) tiene q_vector inválido para cálculo de flujo.")

    logger.debug(f"Flujo calculado (suma de componente {flux_component_index}): {total_flux:.3f}")
    return total_flux

# --- Funciones para Comunicación Bidireccional ---
def fetch_and_apply_torus_field():
    """Obtiene el campo vectorial completo de matriz_ecu y lo aplica a la malla."""
    ecu_vector_field_url = f"{MATRIZ_ECU_BASE_URL}/api/ecu/field_vector"
    logger.debug(f"Intentando obtener campo vectorial de ECU en {ecu_vector_field_url}") # Log de intento
    try:
        response = requests.get(ecu_vector_field_url, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status() # Lanza HTTPError para respuestas 4xx o 5xx
        
        response_data_dict = response.json() # response_data_dict es ahora el objeto JSON completo

        # Verificar que la respuesta es un diccionario y tiene el estado y la clave esperados
        if isinstance(response_data_dict, dict) and \
           response_data_dict.get("status") == "success" and \
           "field_vector" in response_data_dict:
            
            actual_field_vector_list = response_data_dict["field_vector"] # Extraer la lista del campo

            # Ahora verificar que el campo extraído es una lista válida
            if isinstance(actual_field_vector_list, list): # No es necesario 'and actual_field_vector_list' aquí,
                                                            # apply_external_field_to_mesh puede manejar una lista vacía si es necesario.
                                                            # O puedes añadirlo si quieres que un campo vacío también sea un error.
                mesh = malla_cilindrica_global
                if mesh:
                    if mesh.cells: # Solo aplicar si la malla tiene celdas
                        logger.debug(f"Aplicando campo vectorial de ECU (primer elemento: {actual_field_vector_list[0] if actual_field_vector_list else 'vacío'}) a la malla.")
                        apply_external_field_to_mesh(mesh, actual_field_vector_list)
                        logger.info("Campo vectorial toroidal (V) obtenido y aplicado exitosamente a la malla.") # Cambiado a INFO para éxito
                    else:
                        logger.warning("Malla global no tiene celdas. No se aplicará el campo vectorial obtenido.")
                else:
                     logger.warning("Malla global no inicializada. No se pudo aplicar el campo vectorial obtenido.")
            else:
                logger.error(f"El contenido de 'field_vector' en la respuesta de {ecu_vector_field_url} no es una lista. Recibido: {type(actual_field_vector_list)}")
        else:
            logger.error(f"Respuesta JSON inválida, no exitosa, o 'field_vector' ausente de {ecu_vector_field_url}. Respuesta: {response_data_dict}")

    except requests.exceptions.Timeout:
        logger.error(f"Timeout al intentar obtener campo vectorial de {ecu_vector_field_url}")
    except requests.exceptions.RequestException as e:
        # Esto capturará errores HTTP si raise_for_status() los lanza, y errores de conexión.
        logger.error(f"Error de red o HTTP al obtener campo vectorial de {ecu_vector_field_url}: {e}")
    except (ValueError, TypeError, json.JSONDecodeError) as e: # json.JSONDecodeError es más específico
         logger.error(f"Error al procesar/decodificar respuesta JSON de {ecu_vector_field_url}: {e}")
    except Exception as e:
        logger.exception(f"Error inesperado al obtener/aplicar campo vectorial de {ecu_vector_field_url}")

def map_cylinder_to_torus_coords(cell: Cell) -> Optional[Tuple[int, int, int]]:
    """
    Mapea coordenadas y estado de una celda del cilindro a coordenadas del toroide (capa, row, col).
    Retorna None si el mapeo no es posible o las dimensiones del toroide son inválidas.
    """
    # Verificar si la malla global está inicializada antes de intentar mapear
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
         logger.warning("Malla no inicializada o vacía. Saltando mapeo a toroide.")
         return None

    if TORUS_NUM_CAPAS <= 0 or TORUS_NUM_FILAS <= 0 or TORUS_NUM_COLUMNAS <= 0:
        logger.error("Dimensiones del toroide inválidas para mapeo.")
        return None

    # Mapeo de theta a columna del toroide (0 a 2*pi -> 0 a num_cols)
    col_float = (cell.theta / (2 * math.pi)) * TORUS_NUM_COLUMNAS
    col = int(math.floor(col_float)) % TORUS_NUM_COLUMNAS # Usar módulo para periodicidad

    # Mapeo de Z a fila del toroide (min_z a max_z -> 0 a num_rows-1)
    cylinder_height = mesh.max_z - mesh.min_z
    row = 0
    if cylinder_height > EPSILON:
        normalized_z = (cell.z - mesh.min_z) / cylinder_height
        row_float = normalized_z * (TORUS_NUM_FILAS - 1)
        row = int(round(row_float))
        row = max(0, min(row, TORUS_NUM_FILAS - 1)) # Clamp row index
    elif TORUS_NUM_FILAS > 0: # Si la altura es 0 o muy pequeña, mapear todas a la primera fila
         row = 0
    else: # No hay filas en el toroide
         return None

    # Usar la MAGNITUD de la actividad (sqrt(amp^2 + vel^2)) para mapear a la capa
    activity_magnitude = math.sqrt(cell.amplitude**2 + cell.velocity**2)
    normalized_activity = min(1.0, max(0.0, activity_magnitude / MAX_AMPLITUDE_FOR_NORMALIZATION))

    # Mayor actividad -> menor índice de capa (capas "más internas" o "críticas")
    capa = int(round((1.0 - normalized_activity) * (TORUS_NUM_CAPAS - 1)))
    capa = max(0, min(capa, TORUS_NUM_CAPAS - 1)) # Clamp capa index

    return capa, row, col

def send_influence_to_torus(dphi_dt: float):
    """
    Envía una influencia a matriz_ecu basada en la tasa de cambio del flujo magnético (dPhi/dt).
    La influencia se aplica a una ubicación fija o representativa en el toroide.
    """
    # Definir una ubicación fija o representativa en el toroide para aplicar la influencia
    # Podría ser el centro conceptual, o una ubicación relacionada con el estado general de la malla.
    # Usemos una ubicación fija por simplicidad inicial.
    target_capa = 0 # Capa más interna/crítica
    target_row = TORUS_NUM_FILAS // 2 # Fila central
    target_col = TORUS_NUM_COLUMNAS // 2 # Columna central

    # El vector de influencia enviado a ECU se deriva de dPhi/dt.
    # Podría ser [dphi_dt, 0.0], o [sign(dphi_dt), abs(dphi_dt)_normalizado], etc.
    # Usemos [dphi_dt, 0.0] por ahora. ECU interpretará este vector.
    influence_vector = [dphi_dt, 0.0]
    watcher_name = f"malla_watcher_dPhiDt{dphi_dt:.3f}" # Nombre basado en dPhi/dt

    payload = {
        "capa": target_capa,
        "row": target_row,
        "col": target_col,
        "vector": influence_vector,
        "nombre_watcher": watcher_name
    }

    ecu_influence_url = f"{MATRIZ_ECU_BASE_URL}/api/ecu/influence"
    try:
        response = requests.post(ecu_influence_url, json=payload, timeout=REQUESTS_TIMEOUT)
        response.raise_for_status()
        logger.info(f"Influencia (dPhi/dt) enviada a {ecu_influence_url} en ({target_capa}, {target_row}, {target_col}). Payload: {payload}. Respuesta: {response.status_code}")
    except requests.exceptions.Timeout:
        logger.error(f"Timeout al enviar influencia (dPhi/dt) a {ecu_influence_url}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de red al enviar influencia (dPhi/dt) a {ecu_influence_url}: {e}")
    except Exception as e:
        logger.exception(f"Error inesperado al enviar influencia (dPhi/dt) a {ecu_influence_url}")

# --- Bucle de Simulación (Thread) ---
SIMULATION_INTERVAL = float(os.environ.get("MW_SIM_INTERVAL", 0.5)) # Segundos (esto es dt)
DPHI_DT_INFLUENCE_THRESHOLD = float(os.environ.get("MW_DPHI_DT_THRESHOLD", 1.0)) # Umbral de |dPhi/dt|

simulation_thread = None
stop_simulation_event = threading.Event()

# --- Bucle de Simulación (Thread) ---
SIMULATION_INTERVAL = float(os.environ.get("MW_SIM_INTERVAL", 0.5)) # Segundos (esto es dt)
# NUEVO: Umbral para enviar influencia basado en dPhi/dt
DPHI_DT_INFLUENCE_THRESHOLD = float(os.environ.get("MW_DPHI_DT_THRESHOLD", 1.0)) # Umbral de |dPhi/dt|

simulation_thread = None
stop_simulation_event = threading.Event()

def simulation_loop():
    logger.info("Iniciando bucle de simulación de malla...")
    step_count = 0
    dt = SIMULATION_INTERVAL # El paso de tiempo es el intervalo de simulación

    # Acceder directamente a la malla global (que puede ser None o un mock en tests)
    mesh = malla_cilindrica_global
    if mesh is None:
         logger.error("Malla global no inicializada. No se puede iniciar el bucle de simulación.")
         return # Salir si la malla no está inicializada

    # Asegurarse de que el flujo previo está inicializado
    if not hasattr(mesh, 'previous_flux'):
         mesh.previous_flux = 0.0
         logger.warning("previous_flux no inicializado en malla_cilindrica_global. Inicializando a 0.0")

    while not stop_simulation_event.is_set():
        start_time = time.monotonic()
        step_count += 1
        logger.debug(f"--- Iniciando paso de simulación {step_count} ---")

        try:
            # 1. Obtener el campo vectorial completo de ECU y aplicarlo a la malla
            logger.debug(f"Paso {step_count}: Obteniendo campo vectorial del toroide...")
            fetch_and_apply_torus_field() # Esta función maneja si la malla global es None

            # 2. Calcular el flujo magnético actual a través de la malla
            logger.debug(f"Paso {step_count}: Calculando flujo magnético...")
            # Pasar la instancia de la malla a calculate_flux
            current_flux = calculate_flux(mesh)

            # 3. Calcular la tasa de cambio del flujo (dPhi/dt)
            dphi_dt = (current_flux - mesh.previous_flux) / dt if dt > 0 else 0.0
            mesh.previous_flux = current_flux # Actualizar flujo previo

            logger.debug(f"Paso {step_count}: Flujo actual={current_flux:.3f}, dPhi/dt={dphi_dt:.3f}")

            # 4. Simular la dinámica interna de la malla (osciladores acoplados)
            # Nota: La dinámica usa el campo vectorial (q_vector) aplicado en el paso 1
            logger.debug(f"Paso {step_count}: Simulando dinámica interna de la malla...")
            simular_paso_malla() # Esta función accede a la malla global

            # 5. Actualizar estado agregado (basado en magnitud de amplitud)
            logger.debug(f"Paso {step_count}: Actualizando estado agregado...")
            update_aggregate_state() # Esta función accede a la malla global

            # 6. Verificar y enviar influencias al toroide (basado en dPhi/dt)
            # Enviar influencia si la MAGNITUD de dPhi/dt supera un umbral
            if abs(dphi_dt) > DPHI_DT_INFLUENCE_THRESHOLD:
                logger.info(f"Paso {step_count}: |dPhi/dt|={abs(dphi_dt):.3f} supera umbral {DPHI_DT_INFLUENCE_THRESHOLD}. Enviando influencia...")
                send_influence_to_torus(dphi_dt) # Enviar dPhi/dt como base de la influencia
            else:
                 logger.debug(f"Paso {step_count}: |dPhi/dt|={abs(dphi_dt):.3f} no supera umbral para influenciar toroide.")

        except Exception as e:
            logger.exception(f"Error durante el paso de simulación {step_count} de malla.")

        elapsed_time = time.monotonic() - start_time
        sleep_time = max(0, dt - elapsed_time)
        logger.debug(f"--- Paso {step_count} completado en {elapsed_time:.3f}s. Durmiendo por {sleep_time:.3f}s ---")
        if sleep_time > 0:
            stop_simulation_event.wait(sleep_time)

    logger.info("Bucle de simulación de malla detenido.")

AGENT_AI_REGISTER_URL = os.environ.get("AGENT_AI_REGISTER_URL", "http://agent_ai:9000/api/register")
MAX_REGISTRATION_RETRIES = 5 # Intentar registrarse varias veces al inicio
RETRY_DELAY = 5 # Segundos entre reintentos

def register_with_agent_ai(module_name: str, module_url: str, health_url: str, module_type: str, aporta_a: str, naturaleza: str, description: str = ""):
    """Intenta registrar este módulo con AgentAI, con reintentos."""
    payload = {
        "nombre": module_name,
        "url": module_url, # URL base para control/estado
        "url_salud": health_url, # URL específica de salud
        "tipo": module_type,
        "aporta_a": aporta_a,
        "naturaleza_auxiliar": naturaleza,
        "descripcion": description
        # Podrías añadir más metadata si AgentAI la usa
    }
    logger.info(f"Intentando registrar '{module_name}' en AgentAI ({AGENT_AI_REGISTER_URL})...")
    for attempt in range(MAX_REGISTRATION_RETRIES):
        try:
            response = requests.post(AGENT_AI_REGISTER_URL, json=payload, timeout=4.0)
            response.raise_for_status() # Lanza excepción para errores 4xx/5xx
            if response.status_code == 200:
                 logger.info(f"Registro de '{module_name}' exitoso en AgentAI.")
                 return True # Salir si el registro es exitoso
            else:
                 # Esto no debería ocurrir si raise_for_status funciona, pero por si acaso
                 logger.warning(f"Registro de '{module_name}' recibido con status {response.status_code}. Respuesta: {response.text}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión al intentar registrar '{module_name}' (intento {attempt + 1}/{MAX_REGISTRATION_RETRIES}): {e}")
        except Exception as e:
             logger.error(f"Error inesperado durante el registro de '{module_name}' (intento {attempt + 1}/{MAX_REGISTRATION_RETRIES}): {e}")

        if attempt < MAX_REGISTRATION_RETRIES - 1:
            logger.info(f"Reintentando registro en {RETRY_DELAY} segundos...")
            time.sleep(RETRY_DELAY)
        else:
            logger.error(f"No se pudo registrar '{module_name}' en AgentAI después de {MAX_REGISTRATION_RETRIES} intentos.")
            return False # Falló después de todos los reintentos
    return False # En caso de que el bucle termine inesperadamente

# --- Servidor Flask ---
app = Flask(__name__)

# --- Endpoints de Flask ---

# ------------------ ENDPOINT /api/health ------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    sim_alive = simulation_thread.is_alive() if simulation_thread else False
    # Acceder directamente a la malla global (que puede ser None o un mock en tests)
    mesh = malla_cilindrica_global
    num_cells = len(mesh.cells) if mesh and mesh.cells else 0
    status = "success" # Default
    message = "Malla_watcher operativo." # Default

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
        status = "error" # O "warning" si el hilo caído no es crítico
        message = "Error: Hilo de simulación del resonador inactivo."
    else: # Malla inicializada, no vacía, sim activa. Verificar conectividad.
         try:
             # Llamar a verify_connectivity en la instancia global (real o mock)
             connectivity_counts = mesh.verify_connectivity() # Esto loguea internamente
             if connectivity_counts:
                 min_neighbors = min(connectivity_counts.keys())
                 max_neighbors = max(connectivity_counts.keys())
                 if max_neighbors > 6 or (min_neighbors < 3 and num_cells > 1):
                      connectivity_status = "error"
                      status = "error"
                      message = "Error estructural: Problemas graves de conectividad en la malla."
                 elif min_neighbors < 6 and num_cells > 1:
                      connectivity_status = "warning"
                      # Si ya es error por sim_alive o num_cells, no degradar el status
                      if status == "success":
                           status = "warning"
                           message = "Advertencia: Posibles problemas de conectividad en la malla."
                 else:
                      connectivity_status = "ok"
             else: # Malla no vacía pero verify_connectivity retornó dict vacío
                  connectivity_status = "warning"
                  if status == "success":
                       status = "warning"
                       message = "Advertencia: verify_connectivity retornó un resultado vacío inesperado."

         except Exception as e:
             logger.error(f"Error durante verify_connectivity en health check: {e}")
             connectivity_status = "error"
             status = "error"
             message = f"Error interno al verificar conectividad: {e}"

    # --- Construir Respuesta JSON Detallada ---
    response_data = {
        "status": status,
        "module": "Malla_watcher",
        "message": message,
        "details": {
            "mesh": {
                # Verificar si mesh no es None antes de acceder a sus atributos
                "initialized": mesh is not None,
                "num_cells": num_cells,
                "connectivity_status": connectivity_status,
                "min_neighbors": min_neighbors,
                "max_neighbors": max_neighbors,
                "z_periodic": mesh.periodic_z if mesh else None # Acceder solo si mesh no es None
            },
            "resonator_simulation": {
                "running": sim_alive,
            }
        }
    }

    # Usar códigos de estado HTTP apropiados
    http_status_code = 200
    if status == "warning":
        http_status_code = 503 # Service Unavailable (con advertencia)
    elif status == "error":
        http_status_code = 500 # Internal Server Error (si es error de inicialización/estructura)
        # O 503 si es un error temporal como hilo caído

    logger.debug(f"Health check response: Status={status}, HTTP={http_status_code}")
    return jsonify(response_data), http_status_code

# --- Endpoint de Estado ---
@app.route('/api/state', methods=['GET'])
def get_malla_state():
    """Devuelve el estado agregado actual de la malla y los parámetros de control."""
    state_data = {}
    with aggregate_state_lock:
        state_data.update(aggregate_state) # Copia el estado agregado (ahora basado en magnitud)

    # Leer parámetros directamente de las instancias globales (más fiable)
    # Acceder directamente a las instancias globales
    with control_lock: # Usar lock por si acaso, aunque la lectura suele ser atómica
        state_data["control_params"] = {
            "phoswave_C": resonador_global.C, # Renombrado
            "electron_D": electron_global.D   # Renombrado
        }

    # Acceder directamente a la malla global (que puede ser None o un mock en tests)
    mesh = malla_cilindrica_global
    state_data["num_cells"] = len(mesh.cells) if mesh and mesh.cells else 0

    # Opcional: Añadir estadísticas sobre velocidad si es relevante
    # all_cells = malla_cilindrica_global.get_all_cells()
    # velocities = [cell.velocity for cell in all_cells]
    # if velocities:
    #     state_data["avg_velocity"] = sum(velocities) / len(velocities)
    #     state_data["max_velocity"] = max(velocities)
    logger.debug(f"Devolviendo estado agregado: {state_data}")
    return jsonify({"status": "success", "state": state_data})

# ------------------ ENDPOINT /api/control ------------------
@app.route('/api/control', methods=['POST'])
def set_malla_control():
    data = request.get_json(silent=True) # Usar silent=True para evitar errores internos de Flask con JSON inválido
    # Validar si data es None O si falta el campo
    if data is None or "control_signal" not in data: # <-- Asegúrate de esta validación
        logger.error("Solicitud a /api/control sin payload JSON válido o 'control_signal'")
        # Asegurarse de que siempre se retorna un JSON válido en caso de error
        return jsonify({"status": "error", "message": "Payload JSON vacío, inválido o falta 'control_signal'"}), 400
    signal = data['control_signal']
    if not isinstance(signal, (int, float)):
        return jsonify({
            "status": "error",
            "message": "El campo 'control_signal' debe ser un número."
        }), 400

    # Acceder directamente a las instancias globales
    with control_lock: # Usar lock al modificar parámetros compartidos
        new_C = max(0.0, BASE_COUPLING_T + K_GAIN_COUPLING * signal)
        new_D = max(0.0, BASE_DAMPING_E - K_GAIN_DAMPING * signal)
        resonador_global.ajustar_coeficientes(new_C) # Usar método de la clase
        electron_global.ajustar_coeficientes(new_D) # Usar método de la clase
        # Actualizar control_params para reflejar el estado actual
        control_params["phoswave_C"] = resonador_global.C
        control_params["electron_D"] = electron_global.D

    logger.info(f"Parámetros de control ajustados: C={resonador_global.C:.3f}, D={electron_global.D:.3f} (señal={signal:.3f})")
    return jsonify({"status": "success", "message": "Parámetros ajustados", "current_params": control_params}), 200

# ------------------ ENDPOINT /api/malla ------------------
@app.route('/api/malla', methods=['GET'])
def get_malla():
    # Acceder directamente a la malla global (que puede ser None o un mock en tests)
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
         return jsonify({
              "status": "error",
              "message": "Malla no inicializada o vacía."
         }), 503 # Service Unavailable si la malla no está lista

    return jsonify({
         "status": "success",
         "metadata": {
            "radius": mesh.radius,
             "num_cells": len(mesh.cells),
             "periodic_z": mesh.periodic_z,
             "z_bounds": {"min": mesh.min_z, "max": mesh.max_z}
        },
        # Llamar al método get_all_cells en la instancia global
        "cells": [cell.to_dict() for cell in mesh.get_all_cells()]
    }), 200

# ----------------Endpoint /api/malla/influence ---------------------
@app.route("/api/malla/influence", methods=["POST"])
def aplicar_influencia_toroide_push():
    """
    Recibe el CAMPO VECTORIAL completo del toroide (matriz_ecu) y lo aplica
    a la malla cilíndrica actualizando los q_vector de las celdas
    usando interpolación bilineal. (Alternativa pasiva a fetch_and_apply_torus_field)

    Espera JSON: {"field_vector": List[List[List[float]]]} (Shape: [capas, filas, columnas, 2])
    """
    # Verificar si la malla global está inicializada antes de aplicar el campo
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
         logger.warning("Malla no inicializada o vacía. Saltando aplicación de influencia (push).")
         return jsonify({
              "status": "warning",
              "message": "Malla no inicializada o vacía. Influencia no aplicada."
         }), 503 # Service Unavailable si la malla no está lista

    logger.warning("Recibida llamada a endpoint pasivo /api/malla/influence. Se recomienda usar el fetch activo.")
    data = request.get_json(silent=True) # Usar silent=True
    # Validar si data es None O si falta el campo
    if data is None or "field_vector" not in data: # <-- CAMBIADO: Esperar "field_vector"
        logger.error("Solicitud POST a /api/malla/influence sin 'field_vector'.")
        # Asegurarse de que siempre se retorna un JSON válido en caso de error
        return jsonify({"status": "error", "message": "Payload JSON vacío, inválido o falta 'field_vector'"}), 400

    field_vector_data = data["field_vector"] # <-- CAMBIADO: Leer "field_vector"

    try:
        # Llamar al método apply_external_field en la instancia global
        # apply_external_field ya valida el formato de la lista/array
        apply_external_field_to_mesh(mesh, field_vector_data) # <-- Pasar los datos directamente

        logger.info("Influencia del campo vectorial toroidal (push) aplicada correctamente a la malla (q_vector actualizado).")
        return jsonify({"status": "success", "message": "Campo vectorial externo (push) aplicado a q_vector de las celdas vía interpolación."}), 200

    except ValueError as ve:
         logger.error(f"Error de valor al procesar campo vectorial externo (push): {ve}")
         return jsonify({"status": "error", "message": f"Error en los datos recibidos (push): {ve}"}), 400
    except Exception as e:
        logger.exception("Error inesperado al aplicar influencia del toroide (push).")
        return jsonify({"status": "error", "message": "Error interno al aplicar campo externo (push)."}), 500

# ------------------ ENDPOINT /api/event ------------------
@app.route('/api/event', methods=['POST'])
def receive_event():
    # Acceder directamente a la malla global (que puede ser None o un mock en tests)
    mesh = malla_cilindrica_global
    if mesh is None or not mesh.cells:
         logger.warning("Malla no inicializada o vacía. Saltando aplicación de evento.")
         return jsonify({
              "status": "warning",
              "message": "Malla no inicializada o vacía. Evento no aplicado."
         }), 503 # Service Unavailable si la malla no está lista

    data = request.get_json(silent=True) # Usar silent=True
    if data is None: # <-- Asegúrate de esta validación
        return jsonify({"status": "error", "message": "Payload JSON vacío o inválido"}), 400

    logger.info(f"Evento recibido: {data}")
    if data.get("type") == "pulse" and "coords" in data and "magnitude" in data:
        q = data["coords"].get("q")
        r = data["coords"].get("r")
        try:
            mag = float(data.get("magnitude", 1.0))
        except (ValueError, TypeError):
            return jsonify({"status": "error", "message": "Magnitud inválida"}), 400
        if q is not None and r is not None:
            # Llamar al método get_cell en la instancia global
            cell = mesh.get_cell(q, r)
            if cell:
                logger.info(f"Celda ({q},{r}) - Velocidad antes: {cell.velocity:.3f}")
                # Aplicar el pulso a la velocidad
                cell.velocity += mag
                logger.info(f"Celda ({q},{r}) - Velocidad después: {cell.velocity:.3f}")
            else:
                logger.warning(f"No se encontró celda ({q},{r}) para aplicar evento.")
                return jsonify({"status": "warning", "message": f"Celda ({q},{r}) no encontrada."}), 404 # Not Found
        else:
             return jsonify({"status": "error", "message": "Coordenadas 'q' o 'r' faltantes o inválidas."}), 400
    else:
         return jsonify({"status": "error", "message": "Tipo de evento no soportado o datos incompletos."}), 400

    return jsonify({"status": "success", "message": "Evento procesado"}), 200

# ------------------ ENDPOINT /api/error ------------------
@app.route("/api/error", methods=["POST"])
def receive_error():
    data = request.get_json(silent=True) # Usar silent=True
    if data is None: # <-- Asegúrate de esta validación
        return jsonify({"status": "error", "message": "Payload JSON vacío o inválido"}), 400
    logger.error(f"Error reportado desde otro servicio: {data}")
    return jsonify({"status": "success", "message": "Error procesado"}), 200

# ------------------ ENDPOINT /api/config ------------------
@app.route('/api/config', methods=['GET'])
def get_config():
    mesh = malla_cilindrica_global
    return jsonify({
        "status": "success",
        "config": {
            "malla_config": {
                "radius": float(os.environ.get("MW_RADIUS", 5.0)),
                "height_segments": int(os.environ.get("MW_HEIGHT_SEG", 6)), # Lee de os.environ
                "circumference_segments_target": int(os.environ.get("MW_CIRCUM_SEG", 12)), # Lee de os.environ
                "circumference_segments_actual": mesh.circumference_segments_actual if mesh else None,
                "hex_size": float(os.environ.get("MW_HEX_SIZE", 1.0)), # Lee de os.environ
                "periodic_z": os.environ.get("MW_PERIODIC_Z", "True").lower() == "true" # Lee de os.environ
            },
            "communication_config": { # Estas usan constantes del módulo, lo cual es bueno para testear
                "matriz_ecu_url": MATRIZ_ECU_BASE_URL,
                "torus_dims": f"{TORUS_NUM_CAPAS}x{TORUS_NUM_FILAS}x{TORUS_NUM_COLUMNAS}",
                "influence_threshold": AMPLITUDE_INFLUENCE_THRESHOLD,
                "max_activity_normalization": MAX_AMPLITUDE_FOR_NORMALIZATION
            },
            "simulation_config": { # Estas usan constantes del módulo
                 "interval": SIMULATION_INTERVAL,
                 "dphi_dt_influence_threshold": DPHI_DT_INFLUENCE_THRESHOLD
            },
            "control_config": { # Estas usan constantes del módulo y estado global
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
    # Este bloque se ejecuta solo cuando el script es ejecutado directamente.
    # Su propósito es configurar y ejecutar el servicio REAL.

    # 1. Leer las variables de configuración REALES desde el entorno
    # Usar los nombres de constantes originales que leen de ENV
    # Estas constantes ya se leen al importar el módulo, pero las re-leemos aquí
    # para asegurar que usamos los valores del entorno en tiempo de ejecución.
    # Sin embargo, las instancias globales (malla_cilindrica_global, resonador_global, electron_global)
    # ya fueron creadas *antes* de este bloque usando los valores de ENV disponibles en ese momento.
    # La forma correcta es RE-INICIALIZAR las instancias globales aquí con los valores de ENV.

    # Re-leer constantes de ENV (redundante si ya se leen arriba, pero clarifica)
    MESH_RADIUS_REAL = float(os.environ.get("MW_RADIUS", 5.0))
    MESH_HEIGHT_SEGMENTS_REAL = int(os.environ.get("MW_HEIGHT_SEG", 6))
    MESH_CIRCUMFERENCE_SEGMENTS_REAL = int(os.environ.get("MW_CIRCUM_SEG", 12))
    MESH_HEX_SIZE_REAL = float(os.environ.get("MW_HEX_SIZE", 1.0))
    MESH_PERIODIC_Z_REAL = os.environ.get("MW_PERIODIC_Z", "True").lower() == "true"

    MATRIZ_ECU_BASE_URL_REAL = os.environ.get("MATRIZ_ECU_URL", "http://ecu:8000")
    TORUS_NUM_CAPAS_REAL = int(os.environ.get("TORUS_NUM_CAPAS", 3))
    TORUS_NUM_FILAS_REAL = int(os.environ.get("TORUS_NUM_FILAS", 4))
    TORUS_NUM_COLUMNAS_REAL = int(os.environ.get("TORUS_NUM_COLUMNAS", 5))
    AMPLITUDE_INFLUENCE_THRESHOLD_REAL = float(os.environ.get("MW_INFLUENCE_THRESHOLD", 5.0))
    MAX_AMPLITUDE_FOR_NORMALIZATION_REAL = float(os.environ.get("MW_MAX_AMPLITUDE_NORM", 20.0))
    REQUESTS_TIMEOUT_REAL = float(os.environ.get("MW_REQUESTS_TIMEOUT", 2.0))
    BASE_COUPLING_T_REAL = float(os.environ.get("MW_BASE_T", 0.6))
    BASE_DAMPING_E_REAL = float(os.environ.get("MW_BASE_E", 0.1))
    K_GAIN_COUPLING_REAL = float(os.environ.get("MW_K_GAIN_T", 0.1))
    K_GAIN_DAMPING_REAL = float(os.environ.get("MW_K_GAIN_E", 0.05))
    SIMULATION_INTERVAL_REAL = float(os.environ.get("MW_SIM_INTERVAL", 0.5))
    DPHI_DT_INFLUENCE_THRESHOLD_REAL = float(os.environ.get("MW_DPHI_DT_THRESHOLD", 1.0))

    # 2. Loguear la configuración REAL
    logger.info(f"Configuración Malla REAL: R={MESH_RADIUS_REAL}, HSeg={MESH_HEIGHT_SEGMENTS_REAL}, CSeg={MESH_CIRCUMFERENCE_SEGMENTS_REAL}, Hex={MESH_HEX_SIZE_REAL}, PerZ={MESH_PERIODIC_Z_REAL}")
    logger.info(f"Configuración Comms REAL: ECU_URL={MATRIZ_ECU_BASE_URL_REAL}, Torus={TORUS_NUM_CAPAS_REAL}x{TORUS_NUM_FILAS_REAL}x{TORUS_NUM_COLUMNAS_REAL}, InfThr={AMPLITUDE_INFLUENCE_THRESHOLD_REAL}")
    logger.info(f"Configuración Control REAL: BaseC={BASE_COUPLING_T_REAL}, BaseD={BASE_DAMPING_E_REAL}, GainC={K_GAIN_COUPLING_REAL}, GainD={K_GAIN_DAMPING_REAL}")
    logger.info(f"Configuración Simulación REAL: Interval={SIMULATION_INTERVAL_REAL}s, dPhi/dt Threshold={DPHI_DT_INFLUENCE_THRESHOLD_REAL}")
    logger.info(f"Configuración Normalización Influencia REAL: MaxActivityNorm={MAX_AMPLITUDE_FOR_NORMALIZATION_REAL}")

    # 3. RE-Inicializar las instancias globales REAL
    # Esto es crucial para usar los valores de ENV leídos en tiempo de ejecución
    # y no los que estaban disponibles cuando el módulo fue importado.

    # Asignar los valores REALES a las constantes globales
    # Estas asignaciones ahora modificarán las variables globales existentes sin necesidad de 'global'
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
    K_GAIN_DAMPING = K_GAIN_DAMPING_REAL # <-- CORREGIDO: Usar K_GAIN_DAMPING_REAL
    SIMULATION_INTERVAL = SIMULATION_INTERVAL_REAL
    DPHI_DT_INFLUENCE_THRESHOLD = DPHI_DT_INFLUENCE_THRESHOLD_REAL

    try:
        logger.info("Intentando RE-inicializar la instancia global REAL de HexCylindricalMesh...")
        malla_cilindrica_global = HexCylindricalMesh(
            radius=MESH_RADIUS_REAL,
            height_segments=MESH_HEIGHT_SEGMENTS_REAL,
            circumference_segments_target=MESH_CIRCUMFERENCE_SEGMENTS_REAL,
            hex_size=MESH_HEX_SIZE_REAL,
            periodic_z=MESH_PERIODIC_Z_REAL
        )
        if not malla_cilindrica_global.cells:
             logger.error("¡La malla REAL se inicializó pero está vacía! No se generaron celdas.")
             # Si la malla está vacía, el servicio no puede operar. Salir.
             exit(1)
        else:
             logger.info(f"Malla REAL inicializada con {len(malla_cilindrica_global.cells)} celdas.")

    except Exception as e:
        logger.exception("Error crítico al RE-inicializar HexCylindricalMesh global REAL. Terminando.")
        exit(1) # Salir si la malla no se puede inicializar

    # Re-inicializar las instancias globales de PhosWave y Electron con los valores REALES
    resonador_global = PhosWave(coef_acoplamiento=BASE_COUPLING_T)
    electron_global = Electron(coef_amortiguacion=BASE_DAMPING_E)

    # Asegurarse de que previous_flux está inicializado en la instancia REAL
    # El constructor ya lo hace, pero esta línea es una doble verificación si se desea
    if not hasattr(malla_cilindrica_global, 'previous_flux'): # Esta condición siempre será falsa ahora
         malla_cilindrica_global.previous_flux = 0.0
         logger.info("previous_flux inicializado a 0.0 en instancia REAL.")
    # Simplificamos, ya que el constructor lo maneja:
    logger.debug(f"previous_flux en instancia REAL: {malla_cilindrica_global.previous_flux}")

    # Actualizar control_params con los valores REALES iniciales
    with control_lock:
        control_params["phoswave_C"] = resonador_global.C
        control_params["electron_D"] = electron_global.D
    logger.info(f"Parámetros de control iniciales REALES: C={control_params['phoswave_C']:.3f}, D={control_params['electron_D']:.3f}")

    # 4. Registro con AgentAI
    MODULE_NAME = "malla_watcher"
    SERVICE_PORT = int(os.environ.get("PORT", 5001))
    MODULE_URL = f"http://{MODULE_NAME}:{SERVICE_PORT}"
    HEALTH_URL = f"{MODULE_URL}/api/health"
    APORTA_A = "matriz_ecu"
    NATURALEZA = "modulador"
    DESCRIPTION = "Simulador de malla hexagonal cilíndrica (osciladores acoplados) acoplado a ECU, influye basado en inducción electromagnética." # Actualizar descripción

    # Solo intentar registrar si la malla se inicializó correctamente y tiene celdas
    # (Ya salimos con exit(1) si no fue así)
    registration_successful = register_with_agent_ai(
        MODULE_NAME, MODULE_URL, HEALTH_URL, "auxiliar", APORTA_A, NATURALEZA, DESCRIPTION
    )
    if not registration_successful:
        logger.warning(f"El módulo '{MODULE_NAME}' continuará sin registro exitoso en AgentAI.")

    # 5. Iniciar Hilo de Simulación y Servidor Flask
    # Solo iniciar si la malla global se inicializó correctamente Y tiene celdas
    # (Ya salimos con exit(1) si no fue así)
    logger.info("Creando e iniciando hilo de simulación de malla...")
    stop_simulation_event.clear()
    # Pasar el intervalo de simulación REAL al hilo
    simulation_thread = threading.Thread(target=simulation_loop, daemon=True, name="MallaSimLoop") # simulation_loop lee SIMULATION_INTERVAL globalmente
    simulation_thread.start()
    logger.info("Hilo de simulación iniciado.")

    logger.info(f"Iniciando servicio Flask de malla_watcher en puerto {SERVICE_PORT}")
    # Usar un servidor WSGI de producción como waitress o gunicorn en lugar de app.run para producción
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=SERVICE_PORT)
    app.run(host="0.0.0.0", port=SERVICE_PORT, debug=False, use_reloader=False) # Mantener para desarrollo/pruebas

    # Código de limpieza al detener el servidor (ej. con CTRL+C)
    logger.info("Señal de detención recibida. Deteniendo hilo de simulación...")
    stop_simulation_event.set()
    if simulation_thread:
        # Darle tiempo al hilo para terminar limpiamente
        simulation_thread.join(timeout=SIMULATION_INTERVAL * 3 + 5) # Dar un poco más de tiempo
        if simulation_thread.is_alive():
            logger.warning("El hilo de simulación no terminó a tiempo.")
    logger.info("Servicio malla_watcher finalizado.")

# --- END OF FILE malla_watcher.py (FINAL MODIFICADO) ---