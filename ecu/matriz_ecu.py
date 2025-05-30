# --- START OF FILE matriz_ecu.py (CORREGIDO) ---

#!/usr/bin/env python3
"""
Módulo que define la matriz ECU (Experiencia de Campo Unificado) para el sistema Watchers,
inspirada en un campo de confinamiento magnético toroidal (tipo Tokamak).
Modela el estado y la dinámica de los watchers dentro de este campo multicapa.
Proporciona servicios REST para obtener el estado unificado y recibir influencias.
Incluye un hilo de simulación en segundo plano para la evolución dinámica del campo.

El campo es representado como un campo vectorial 2D discreto en una grilla 3D toroidal.
Cada punto de la grilla (capa, fila, columna) almacena un vector [vx, vy],
que puede interpretarse como componentes locales de la densidad de flujo magnético (B).
La interpretación exacta de vx y vy en términos de direcciones físicas (toroidal, poloidal, radial)
se puede definir conceptualmente por analogía. Por ejemplo:
vx podría ser la componente toroidal de B.
vy podría ser la componente poloidal (vertical) de B.
La dimensión de la "capa" podría representar la dirección radial.
"""

import numpy as np
import logging
import os
import time      # Añadido
import threading
from flask import Flask, jsonify, request
from typing import List, Optional, Tuple, Dict, Any

# --- Configuración del Logging ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
if not logging.getLogger("matriz_ecu").hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s", # Añadido threadName
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "matriz_ecu.log")),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger("matriz_ecu")

# --- Constantes Configurables para el Campo Toroidal ---
NUM_CAPAS = int(os.environ.get("ECU_NUM_CAPAS", 3)) # Renombrado ENV var para consistencia
NUM_FILAS = int(os.environ.get("ECU_NUM_FILAS", 4))
NUM_COLUMNAS = int(os.environ.get("ECU_NUM_COLUMNAS", 5))
# Valores default para parámetros por capa (usados si no se proporcionan en __init__)
DEFAULT_ALPHA_VALUE = 0.5
DEFAULT_DAMPING_VALUE = 0.05

# --- NUEVO: Configuración para la Simulación Dinámica ---
SIMULATION_INTERVAL = float(os.environ.get("ECU_SIM_INTERVAL", 1.0)) # Intervalo en segundos (dt)
BETA_COUPLING = float(os.environ.get("ECU_BETA_COUPLING", 0.1)) # Factor de acoplamiento vertical

# --- Clase ToroidalField ---
class ToroidalField:
    """
    Representa un campo de confinamiento magnético toroidal discreto.

    El campo se modela en una grilla 3D (capas x filas x columnas).
    Cada punto de la grilla (capa, fila, columna) almacena un vector 2D [vx, vy],
    interpretado como componentes locales de la densidad de flujo magnético (B).
    La dinámica simula la evolución de este campo bajo efectos de advección/rotación,
    acoplamiento vertical y disipación, inspirada en la física de plasmas confinados.
    """
    def __init__(self, num_capas: int, num_rows: int, num_cols: int,
                 alphas: Optional[List[float]] = None,
                 dampings: Optional[List[float]] = None):
        """
        Inicializa el campo toroidal con dimensiones y parámetros por capa.

        Args:
            num_capas (int): Número de capas (dimensión radial/profundidad).
            num_rows (int): Número de filas (dimensión poloidal/vertical).
            num_cols (int): Número de columnas (dimensión toroidal/azimutal).
            alphas (Optional[List[float]]): Coeficientes de advección/rotación por capa.
                                            Relacionado con la dinámica toroidal del campo.
            dampings (Optional[List[float]]): Coeficientes de disipación/amortiguación por capa.
                                             Relacionado con la pérdida de intensidad del campo local.
        """
        if num_capas <= 0 or num_rows <= 0 or num_cols <= 0:
            logger.error("Las dimensiones (capas, filas, columnas) deben ser positivas.")
            raise ValueError("Las dimensiones (capas, filas, columnas) deben ser positivas.")

        self.num_capas = num_capas
        self.num_rows = num_rows
        self.num_cols = num_cols
        # self.campo almacena el campo vectorial 2D en cada punto de la grilla 3D
        # Shape: [num_capas, num_rows, num_cols, 2]
        self.campo = [np.zeros((self.num_rows, self.num_cols, 2)) for _ in range(self.num_capas)]
        self.lock = threading.Lock()

        # --- Definir valores default locales ---
        local_default_alpha = DEFAULT_ALPHA_VALUE
        local_default_damping = DEFAULT_DAMPING_VALUE

        # --- Configuración de parámetros por capa ---
        if alphas is None:
            self.alphas = [local_default_alpha] * self.num_capas
            logger.info(f"Usando alpha por defecto: {self.alphas}")
        elif len(alphas) == self.num_capas:
            self.alphas = alphas
            logger.info(f"Usando alphas proporcionados: {self.alphas}")
        else:
            logger.error(f"La lista 'alphas' debe tener longitud {self.num_capas}, pero tiene {len(alphas)}.")
            raise ValueError(f"La lista 'alphas' debe tener longitud {self.num_capas}.")

        if dampings is None:
            self.dampings = [local_default_damping] * self.num_capas
            logger.info(f"Usando damping por defecto: {self.dampings}")
        elif len(dampings) == self.num_capas:
            self.dampings = dampings
            logger.info(f"Usando dampings proporcionados: {self.dampings}")
        else:
            logger.error(f"La lista 'dampings' debe tener longitud {self.num_capas}, pero tiene {len(dampings)}.")
            raise ValueError(f"La lista 'dampings' debe tener longitud {self.num_capas}.")

        logger.info(f"Campo toroidal inicializado: {self.num_capas} capas, dims={self.num_rows}x{self.num_cols}")

    def aplicar_influencia(self, capa: int, row: int, col: int, vector: np.ndarray, nombre_watcher: str) -> bool:
        """
        Aplica una influencia externa (vector 2D) a un punto específico del campo.

        Esto simula una perturbación localizada o una inyección de energía/flujo
        en la grilla toroidal por parte de un watcher.

        Args:
            capa (int): Índice de la capa (0 a num_capas-1).
            row (int): Índice de la fila (0 a num_rows-1).
            col (int): Índice de la columna (0 a num_cols-1).
            vector (np.ndarray): Vector 2D [vx, vy] a añadir al campo local.
            nombre_watcher (str): Nombre del watcher que aplica la influencia.

        Returns:
            bool: True si la influencia se aplicó correctamente, False en caso contrario.
        """
        # ... (Validaciones iniciales sin cambios) ...
        if not (0 <= capa < self.num_capas):
            logger.error(f"Índice de capa fuera de rango ({capa}) al aplicar influencia de {nombre_watcher}.")
            return False
        if not (0 <= row < self.num_rows):
            logger.error(f"Índice de fila fuera de rango ({row}) al aplicar influencia de {nombre_watcher}.")
            return False
        if not (0 <= col < self.num_cols):
            logger.error(f"Índice de columna fuera de rango ({col}) al aplicar influencia de {nombre_watcher}.")
            return False
        if not isinstance(vector, np.ndarray) or vector.shape != (2,):
             logger.error(f"Vector de influencia inválido para {nombre_watcher}: {vector}. Debe ser NumPy array (2,).")
             return False
        if not isinstance(nombre_watcher, str) or not nombre_watcher:
             logger.error(f"Nombre de watcher inválido: {nombre_watcher}. Debe ser un string no vacío.")
             return False

        try:
            vector_valido = np.nan_to_num(vector)
            with self.lock:
                self.campo[capa][row, col] += vector_valido
                valor_actual = self.campo[capa][row, col]
            logger.info(f"'{nombre_watcher}' aplicó influencia al campo local en capa {capa}, nodo ({row}, {col}): {vector_valido}. "
                        f"Nuevo valor del campo: {valor_actual}")
            return True
        except Exception as e:
            logger.exception(f"Error inesperado al aplicar influencia de '{nombre_watcher}' en ({capa}, {row}, {col}): {e}")
            return False

    def get_neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        """
        Obtiene las coordenadas (row, col) de los 4 vecinos directos
        (arriba, abajo, izquierda, derecha) de un nodo, aplicando conectividad toroidal
        en la dimensión de columna.
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
        Calcula el gradiente adaptativo (diferencia de magnitud) entre capas de confinamiento adyacentes.
        Puede interpretarse como una medida de la "tensión" o "shear" del campo entre capas.
        """
        if self.num_capas < 2:
            logger.warning("Se necesita al menos 2 capas para calcular el gradiente de confinamiento.")
            return np.array([])

        with self.lock:
            campo_copia = [np.copy(capa) for capa in self.campo]

        gradiente_entre_capas = np.zeros((self.num_capas - 1, self.num_rows, self.num_cols))
        for i in range(self.num_capas - 1):
            diferencia_vectorial = campo_copia[i] - campo_copia[i + 1]
            magnitud_diferencia = np.linalg.norm(diferencia_vectorial, axis=2)
            gradiente_entre_capas[i] = magnitud_diferencia
        logger.debug(f"Gradiente de confinamiento entre capas calculado (shape: {gradiente_entre_capas.shape})")
        return gradiente_entre_capas

    def obtener_campo_unificado(self) -> np.ndarray:
        """
        Genera un mapa escalar unificado (num_rows x num_cols) ponderado por capa.

        Este mapa representa la intensidad o magnitud agregada del campo vectorial
        en cada ubicación (fila, columna) a través de las capas. Puede interpretarse
        como una medida de la "densidad de energía" o "turbulencia" local del campo.
        Las capas más internas (índices bajos) suelen tener mayor peso.

        Returns:
            np.ndarray: Array 2D (num_rows x num_cols) del campo unificado.
        """
        if self.num_capas == 1:
            pesos = np.array([1.0])
        else:
            pesos = np.linspace(1.0, 0.5, self.num_capas)
        logger.debug(f"Pesos por capa para intensidad unificada: {pesos}")

        campo_unificado = np.zeros((self.num_rows, self.num_cols))
        with self.lock:
            campo_copia = [np.copy(capa) for capa in self.campo]

        for i, capa_actual in enumerate(campo_copia):
            magnitud_capa = np.linalg.norm(capa_actual, axis=2)
            campo_unificado += pesos[i] * magnitud_capa

        logger.debug(f"Mapa de intensidad unificada calculado (shape: {campo_unificado.shape})")
        return campo_unificado

    def apply_rotational_step(self, dt: float, beta: float):
        """
        Aplica un paso de simulación (método numérico discreto) para la dinámica
        del campo vectorial toroidal.

        Esta dinámica simula:
        1. Advección/Rotación en la dirección toroidal (columnas), controlada por `alpha` (por capa).
        2. Acoplamiento/Transporte vertical (entre filas), controlado por `beta`.
        3. Disipación/Decaimiento local, controlada por `damping` (por capa).

        Aproxima la evolución del campo vectorial V según una ecuación tipo:
        dV/dt ≈ - alpha * dV/d(toroidal_angle) + beta_up * V_up + beta_down * V_down - damping * V
        (Esta es una simplificación conceptual de la PDE subyacente).

        Args:
            dt (float): Paso de tiempo de la simulación.
            beta (float): Coeficiente de acoplamiento vertical global.
        """
        if beta < 0:
            logger.warning("El factor beta (acoplamiento vertical) debería ser no negativo.")

        with self.lock:
            # Crear una copia para almacenar el estado del siguiente paso
            next_campo = [np.copy(capa) for capa in self.campo]

            for capa_idx in range(self.num_capas):
                alpha_capa = self.alphas[capa_idx]
                damping_capa = self.dampings[capa_idx]

                if alpha_capa < 0 or damping_capa < 0:
                     logger.warning(f"Alpha ({alpha_capa}) o Damping ({damping_capa}) negativos para capa {capa_idx}.")

                for r in range(self.num_rows):
                    for c in range(self.num_cols):
                        v_current = self.campo[capa_idx][r, c] # Campo vectorial actual en (capa, r, c)

                        # --- Calcular influencias de vecinos (términos de advección/acoplamiento) ---

                        # Influencia desde la izquierda (dirección toroidal)
                        # Simula el arrastre o rotación del campo
                        left_c = (c - 1) % self.num_cols # Conectividad toroidal
                        v_left = self.campo[capa_idx][r, left_c]
                        influence_from_left = alpha_capa * v_left * dt # Proporcional al campo vecino y alpha

                        # Influencia desde arriba (dirección poloidal/vertical)
                        # Simula el acoplamiento o transporte vertical
                        up_r = (r - 1) % self.num_rows # Conectividad poloidal (si aplica, aquí es solo acoplamiento)
                        v_up = self.campo[capa_idx][up_r, c]
                        influence_from_up = beta * v_up * dt # Proporcional al campo vecino y beta

                        # Influencia desde abajo (dirección poloidal/vertical)
                        # Simula el acoplamiento o transporte vertical
                        down_r = (r + 1) % self.num_rows # Conectividad poloidal
                        v_down = self.campo[capa_idx][down_r, c]
                        influence_from_down = beta * v_down * dt # Proporcional al campo vecino y beta

                        # --- Calcular disipación local ---
                        # Simula la pérdida de intensidad del campo local
                        v_current_damped = v_current * (1.0 - damping_capa * dt) # Decaimiento exponencial discreto

                        # --- Actualizar el campo en el siguiente paso ---
                        # El nuevo campo es el campo disipado más las influencias de los vecinos
                        next_campo[capa_idx][r, c] = (v_current_damped +
                                                      influence_from_left +
                                                      influence_from_up +
                                                      influence_from_down)

            # Reemplazar el campo actual con el campo calculado para el siguiente paso
            self.campo = next_campo
        logger.debug(f"Paso de dinámica de campo toroidal aplicado con dt={dt}, beta={beta}")


# --- Instancia Global del Campo Toroidal ---
try:
    # Usar las constantes globales para la instancia del servicio
    campo_toroidal_global_servicio = ToroidalField(
        num_capas=NUM_CAPAS,
        num_rows=NUM_FILAS,
        num_cols=NUM_COLUMNAS,
        alphas=None, # Usará los defaults internos basados en num_capas
        dampings=None # Usará los defaults internos basados en num_capas
    )
    logger.info("Aplicando influencias iniciales al campo de confinamiento global (servicio)...")
    campo_toroidal_global_servicio.aplicar_influencia(capa=0, row=1, col=2, vector=np.array([1.0, 0.5]), nombre_watcher="watcher_init_1")
    campo_toroidal_global_servicio.aplicar_influencia(capa=2, row=3, col=0, vector=np.array([0.2, -0.1]), nombre_watcher="watcher_init_2")
    logger.info("Influencias iniciales aplicadas al campo global (servicio).")

except ValueError as e:
    logger.exception("Error crítico al inicializar ToroidalField global (servicio). Terminando.")
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
    while not stop_simulation_event.is_set():
        start_time = time.monotonic()
        try:
            campo_toroidal_global_servicio.apply_rotational_step(dt, beta)
        except Exception as e:
            logger.exception("Error inesperado durante el paso de simulación ECU.")

        elapsed_time = time.monotonic() - start_time
        sleep_time = max(0, dt - elapsed_time)
        if sleep_time < 0.01:
             logger.warning(f"El ciclo de simulación ECU ({elapsed_time:.3f}s) excedió el intervalo ({dt}s).")

        stop_simulation_event.wait(sleep_time)

    logger.info("Bucle de simulación ECU detenido.")


# --- Servidor Flask ---
app = Flask(__name__)

# ... (Endpoints /api/health, /api/ecu, /api/ecu/influence sin cambios funcionales) ...
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Verifica la salud del servicio ECU (Experiencia de Campo Unificado).

    Incluye el estado de inicialización del campo toroidal y el hilo de simulación.
    """
    sim_alive = simulation_thread.is_alive() if simulation_thread else False
    service_ready = campo_toroidal_global_servicio and hasattr(campo_toroidal_global_servicio, 'num_capas')
    status_code = 503 # Service Unavailable por defecto
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
        response["message"] = "Servicio ECU inicializado pero la simulación no está activa."
        status_code = 503 # O 200 con warning, depende de la criticidad
    elif not service_ready:
         response["message"] = "Error: Objeto ToroidalField no inicializado."
         status_code = 500 # Internal server error

    logger.debug(f"Health check: {response}")
    return jsonify(response), status_code

@app.route("/api/ecu", methods=["GET"])
def obtener_estado_unificado_api() -> Tuple[Any, int]:
    """
    Endpoint REST para obtener el mapa escalar unificado del campo toroidal.

    Retorna un array 2D (filas x columnas) que representa la intensidad agregada
    del campo vectorial en cada punto, ponderada por capa.
    """
    try:
        campo_unificado_actual = campo_toroidal_global_servicio.obtener_campo_unificado()
        estado_lista = campo_unificado_actual.tolist()

        logger.info(f"Solicitud GET /api/ecu recibida. Devolviendo mapa de intensidad unificada.")
        response_data = {
            "status": "success",
            "estado_campo_unificado": estado_lista,
            "metadata": {
                "descripcion": "Mapa escalar de intensidad del campo de confinamiento toroidal ponderado por capa (Analogía Tokamak)",
                "capas": campo_toroidal_global_servicio.num_capas,
                "filas": campo_toroidal_global_servicio.num_rows,
                "columnas": campo_toroidal_global_servicio.num_cols
            }
        }
        return jsonify(response_data), 200
    except Exception as e:
        logger.exception(f"Error en endpoint /api/ecu")
        error_response = {
            "status": "error",
            "message": "Error interno al procesar la solicitud de la matriz ECU."
        }
        return jsonify(error_response), 500

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
            logger.error("No se recibió payload JSON en la solicitud POST /api/ecu/influence.")
            return jsonify({"status": "error", "message": "Payload JSON vacío o ausente"}), 400
    except Exception as e:
        logger.error(f"Error al parsear JSON en la solicitud POST /api/ecu/influence: {e}")
        return jsonify({"status": "error", "message": "Error al parsear el payload JSON."}), 400

    required_fields: Dict[str, type] = {
        "capa": int, "row": int, "col": int, "vector": list, "nombre_watcher": str
    }
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        msg = f"Faltan campos requeridos en el JSON: {', '.join(missing_fields)}"
        logger.error(msg)
        return jsonify({"status": "error", "message": msg}), 400

    type_errors = []
    vector_np = None
    for field, expected_type in required_fields.items():
        # Asegurarse que el campo existe antes de verificar el tipo
        if field in data and not isinstance(data[field], expected_type):
            type_errors.append(f"Campo '{field}' debe ser {expected_type.__name__}, recibido {type(data[field]).__name__}")

    vector_data = data.get("vector")
    if isinstance(vector_data, list) and len(vector_data) == 2:
        try:
            vector_np = np.array([float(v) for v in vector_data], dtype=float)
            if vector_np.shape != (2,): type_errors.append("Campo 'vector' debe ser lista de 2 números.")
        except (ValueError, TypeError): type_errors.append("Campo 'vector' debe contener números.")
    # Solo añadir error si 'vector' existe pero no es válido
    elif 'vector' in data:
         type_errors.append("Campo 'vector' debe ser lista de 2 números")

    if type_errors:
        msg = "errores de tipo en json: " + "; ".join(type_errors).lower()
        logger.error(msg)
        return jsonify({"status": "error", "message": msg}), 400

    # Si llegamos aquí, los campos requeridos existen y tienen tipos básicos correctos
    capa, row, col, nombre_watcher = data['capa'], data['row'], data['col'], data['nombre_watcher']

    try:
        success = campo_toroidal_global_servicio.aplicar_influencia(
            capa=capa, row=row, col=col, vector=vector_np, nombre_watcher=nombre_watcher
        )
        if success:
            logger.info(f"Influencia de '{nombre_watcher}' aplicada exitosamente via API en ({capa}, {row}, {col}).")
            return jsonify({
                "status": "success", "message": f"Influencia de '{nombre_watcher}' aplicada.",
                "applied_to": {"capa": capa, "row": row, "col": col}, "vector": vector_np.tolist()
            }), 200
        else:
            logger.error(f"Fallo al aplicar influencia de '{nombre_watcher}' via API en ({capa}, {row}, {col}) por validación interna.")
            return jsonify({"status": "error", "message": "Error de validación al aplicar influencia."}), 400
    except Exception as e:
        logger.exception(f"Error inesperado en endpoint /api/ecu/influence al aplicar influencia: {e}")
        return jsonify({"status": "error", "message": "Error interno al procesar la influencia."}), 500

# Retorna el campo vectorial completo
@app.route("/api/ecu/field_vector", methods=["GET"])
def get_field_vector_api() -> Tuple[Any, int]:
    """
    Endpoint REST para obtener el campo vectorial completo de la grilla toroidal.

    Retorna una estructura 3D (lista de listas de listas) donde cada elemento
    es un vector 2D [vx, vy]. Shape: [num_capas, num_rows, num_cols, 2].
    """
    try:
        with campo_toroidal_global_servicio.lock:
            # Obtener una copia del campo para evitar modificarlo mientras se serializa
            campo_copia = [np.copy(capa).tolist() for capa in campo_toroidal_global_servicio.campo]

        logger.info(f"Solicitud GET /api/ecu/field_vector recibida. Devolviendo campo vectorial completo.")
        response_data = {
            "status": "success",
            "field_vector": campo_copia, # Retornar la copia como lista de listas
            "metadata": {
                "descripcion": "Campo vectorial 2D en grilla 3D toroidal (Analogía Densidad Flujo Magnético)",
                "capas": campo_toroidal_global_servicio.num_capas,
                "filas": campo_toroidal_global_servicio.num_rows,
                "columnas": campo_toroidal_global_servicio.num_cols,
                "vector_dim": 2
            }
        }
        return jsonify(response_data), 200
    except Exception as e:
        logger.exception(f"Error en endpoint /api/ecu/field_vector")
        error_response = {
            "status": "error",
            "message": "Error interno al procesar la solicitud del campo vectorial."
        }
        return jsonify(error_response), 500

# --- Función Principal y Arranque ---
def main():
    global simulation_thread

    logger.info(f"Configuración ECU: {NUM_CAPAS}x{NUM_FILAS}x{NUM_COLUMNAS}, SimInterval={SIMULATION_INTERVAL}s, Beta={BETA_COUPLING}")

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
    logger.info(f"Iniciando servicio Flask de matriz_ecu en puerto {puerto_servicio}...")
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

# --- END OF FILE matriz_ecu.py (CORREGIDO) ---