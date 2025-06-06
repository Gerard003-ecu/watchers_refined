# watchers/watchers_tools/malla_watcher/utils/cilindro_grafenal.py

import math
import logging
import numpy as np
from collections import deque, Counter # Counter para verify_connectivity
from typing import List, Optional, Dict, Tuple, Any

# --- Configuración del Logging ---
# Es buena práctica que cada módulo tenga su propio logger.
# Si este módulo es usado por otros, ellos pueden configurar el handler y nivel.
logger = logging.getLogger(__name__) # Usar __name__ para el logger del módulo
# Ejemplo de configuración básica si se ejecuta este archivo directamente o para pruebas:
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# Pequeña tolerancia para comparaciones de flotantes
EPSILON = 1e-9

# --- Funciones Auxiliares de Geometría ---
def axial_to_cartesian_flat(q: int, r: int, hex_size: float = 1.0) -> tuple[float, float]:
    """Convierte coordenadas axiales (q, r) a cartesianas (x, y) en una malla hexagonal plana."""
    x = hex_size * (3./2 * q)
    y = hex_size * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
    return x, y

def cartesian_flat_to_cylindrical(x_flat: float, y_flat: float, radius: float) -> tuple[float, float, float]:
    """
    Enrolla las coordenadas cartesianas planas (x, y) en un cilindro de radio 'radius'.
    El eje 'x' plano se convierte en la dirección azimutal (theta).
    El eje 'y' plano se convierte en la dirección axial (z).
    """
    theta = x_flat / radius if radius > EPSILON else 0.0
    z = y_flat
    # Normalizar theta a [0, 2*pi)
    theta = theta % (2 * math.pi)
    if theta < 0: # Asegurar que theta sea positivo
        theta += (2 * math.pi)
    return radius, theta, z

# --- Clase Cell (Coordenadas Cilíndricas + Estado de Oscilador) ---
class Cell:
    """
    Representa una celda (nodo) de la malla hexagonal cilíndrica.
    Almacena coordenadas cilíndricas (r, theta, z), estado local de oscilador
    (amplitud, velocidad) y el valor del campo vectorial externo (q_vector) en su ubicación.
    """
    def __init__(self,
                 cyl_radius: float,
                 cyl_theta: float,
                 cyl_z: float,
                 q_axial: int,
                 r_axial: int,
                 amplitude: float = 0.0,
                 velocity: float = 0.0,
                 q_vector: Optional[np.ndarray] = None):
        """
        Inicializa la celda.

        Args:
            cyl_radius (float): Radio cilíndrico (constante para todas las celdas de una malla).
            cyl_theta (float): Ángulo azimutal (en radianes, [0, 2*pi)).
            cyl_z (float): Altura a lo largo del eje del cilindro.
            q_axial (int): Coordenada axial 'q' original (para identificación/vecindad).
            r_axial (int): Coordenada axial 'r' original (para identificación/vecindad).
            amplitude (float): Amplitud del oscilador.
            velocity (float): Velocidad del oscilador.
            q_vector (Optional[np.ndarray]): Valor del campo vectorial externo [vx, vy] asociado a la celda.
                                            Si es None, se inicializa a [0.0, 0.0].
        """
        self.r: float = cyl_radius
        self.theta: float = cyl_theta # En radianes
        self.z: float = cyl_z
        self.q_axial: int = q_axial # Coordenada axial q
        self.r_axial: int = r_axial # Coordenada axial r

        self.amplitude: float = amplitude
        self.velocity: float = velocity
        
        if q_vector is not None and isinstance(q_vector, np.ndarray) and q_vector.shape == (2,):
            self.q_vector: np.ndarray = q_vector
        else:
            self.q_vector: np.ndarray = np.zeros(2, dtype=float)

    def __repr__(self) -> str:
        q_vec_str = f"[{self.q_vector[0]:.2f}, {self.q_vector[1]:.2f}]"
        return (f"Cell(ax=({self.q_axial},{self.r_axial}), "
                f"cyl=(r={self.r:.2f}, θ={self.theta:.2f}, z={self.z:.2f}), "
                f"amp={self.amplitude:.2f}, vel={self.velocity:.2f}, q_v={q_vec_str})")

    def to_dict(self) -> Dict[str, Any]:
        """Retorna una representación de la celda como diccionario, serializable a JSON."""
        return {
            "axial_coords": {"q": self.q_axial, "r": self.r_axial},
            "cylindrical_coords": {"r": self.r, "theta": self.theta, "z": self.z},
            "amplitude": self.amplitude,
            "velocity": self.velocity,
            "q_vector": self.q_vector.tolist() # Convertir a lista para serialización JSON
        }

# --- Clase HexCylindricalMesh (Gestor de la Malla) ---
class HexCylindricalMesh:
    """
    Gestiona una malla hexagonal cilíndrica, incluyendo su creación,
    validación de conectividad y acceso a sus celdas.
    La malla se construye a partir de un radio, número de segmentos en altura y circunferencia,
    y el tamaño de los hexágonos.
    """
    def __init__(self,
                 radius: float,
                 height_segments: int,
                 circumference_segments_target: int, # Renombrado para claridad
                 hex_size: float = 1.0,
                 periodic_z: bool = False):
        """
        Inicializa la malla hexagonal cilíndrica.

        Args:
            radius (float): Radio del cilindro.
            height_segments (int): Número deseado de "capas" o segmentos de hexágonos en altura (eje Z).
                                   Un valor de 0 o 1 puede generar una malla muy plana.
            circumference_segments_target (int): Número deseado de hexágonos para cerrar la circunferencia.
                                                 Este valor se ajustará para un cierre óptimo.
            hex_size (float): Tamaño característico de los hexágonos (distancia del centro a un vértice).
            periodic_z (bool): Si es True, se aplicarán condiciones de contorno periódicas en la dirección Z
                               al buscar vecinos.
        
        Raises:
            ValueError: Si los parámetros de entrada son inválidos (ej. radio muy pequeño).
        """
        if radius <= 0:
            raise ValueError("El radio del cilindro debe ser positivo.")
        if hex_size <= 0:
            raise ValueError("El tamaño del hexágono debe ser positivo.")
        if radius < hex_size: # Un hexágono no cabría bien
             logger.warning(f"El radio ({radius}) es menor que el tamaño del hexágono ({hex_size}). La malla podría ser degenerada.")
        if height_segments < 0:
            raise ValueError("El número de segmentos de altura no puede ser negativo.")
        if circumference_segments_target < 3: # Mínimo para formar algo parecido a un cilindro
            raise ValueError("Se requieren al menos 3 segmentos de circunferencia objetivo.")

        self.radius: float = radius
        self.height_segments: int = max(0, height_segments) # Asegurar no negativo
        self.hex_size: float = hex_size
        self.periodic_z: bool = periodic_z

        self.circumference: float = 2 * math.pi * self.radius
        
        # Ancho efectivo de un hexágono en la dirección de la circunferencia (q)
        # Un hexágono tiene ancho 1.5*lado si lado = hex_size (distancia centro-vertice).
        # Si hex_size es apotema, el ancho es 2*hex_size / sqrt(3) * 1.5.
        # Asumamos hex_size es distancia centro-vértice. El "ancho" en la dirección q es 1.5 * hex_size.
        # No, el ancho de un hexágono regular (distancia entre caras paralelas opuestas) es sqrt(3)*lado.
        # La distancia entre centros de hexágonos adyacentes en la dirección 'q' es 1.5 * hex_size.
        hex_width_circumferential = self.hex_size * 1.5
        
        if hex_width_circumferential < EPSILON:
            raise ValueError("El tamaño del hexágono es demasiado pequeño, resultando en un ancho circunferencial nulo.")

        # Ajustar el número de segmentos en la circunferencia para que cierre lo mejor posible
        # num_hex_circum_ideal = self.circumference / hex_width_circumferential
        # self.circumference_segments_actual = max(3, round(num_hex_circum_ideal))
        # Usamos el circumference_segments_target directamente como el número de 'columnas' q
        # y la periodicidad se encargará del cierre.
        self.circumference_segments_actual = max(3, circumference_segments_target)

        # La circunferencia real cubierta por este número de segmentos
        self.actual_circumference_covered_by_q_segments = self.circumference_segments_actual * hex_width_circumferential
        
        logger.info(f"Malla Cilíndrica: Radio={self.radius:.2f}, AlturaSeg={self.height_segments}, "
                    f"CircumSegTarget={circumference_segments_target} -> Actual={self.circumference_segments_actual}, "
                    f"HexSize={self.hex_size:.2f}, PeriodicZ={self.periodic_z}")
        if not math.isclose(self.actual_circumference_covered_by_q_segments, self.circumference, rel_tol=0.15) and self.circumference_segments_actual > 3 :
             logger.warning(f"La circunferencia teórica ({self.circumference:.2f}) y la cubierta por los "
                            f"segmentos q ({self.actual_circumference_covered_by_q_segments:.2f}) difieren significativamente. "
                            f"Esto es normal si circumference_segments_target no es un múltiplo ideal.")

        self.cells: Dict[Tuple[int, int], Cell] = {}
        self.min_z: float = 0.0
        self.max_z: float = 0.0
        self.total_height_approx: float = 0.0 # Altura aproximada basada en segmentos

        # Atributo para almacenar el flujo previo, usado por malla_watcher.py
        # Aunque no es estrictamente parte de la estructura, es conveniente tenerlo aquí
        # si la malla es la "dueña" de este estado entre pasos de simulación.
        # Alternativamente, malla_watcher.py podría manejarlo externamente.
        # Por ahora, lo mantenemos aquí para compatibilidad con el código original.
        self.previous_flux: float = 0.0

        self._initialize_mesh()
        if self.cells:
            self._calculate_z_bounds()
            self.verify_connectivity() # Verificar conectividad al final
        else:
            logger.error("¡La inicialización de la malla no generó celdas!")

    def _calculate_z_bounds(self):
        """Calcula los límites Z mínimo y máximo reales de las celdas existentes."""
        if not self.cells:
            self.min_z = 0.0
            self.max_z = 0.0
            logger.warning("Intentando calcular límites Z en malla vacía.")
            return

        z_values = [cell.z for cell in self.cells.values()]
        self.min_z = min(z_values)
        self.max_z = max(z_values)
        logger.info(f"Límites Z calculados: min={self.min_z:.2f}, max={self.max_z:.2f}. "
                    f"Altura real de la malla: {self.max_z - self.min_z:.2f}. "
                    f"Altura aprox. por segmentos: {self.total_height_approx:.2f}")

    def _initialize_mesh(self):
        """
        Crea las celdas de la malla hexagonal cilíndrica usando BFS.
        Las coordenadas (q,r) son axiales. 'q' se enrolla para la circunferencia, 'r' contribuye a la altura Z.
        """
        logger.info(f"Inicializando malla hexagonal cilíndrica (periodic_z={self.periodic_z})...")
        self.cells.clear()

        # Altura de una fila de hexágonos (distancia vertical entre centros de hexágonos en la misma columna q, r vs r+1)
        # En coordenadas axiales, un cambio en 'r' (manteniendo 'q') cambia 'y_flat' en sqrt(3)*hex_size.
        # Un cambio en 'q' (manteniendo 'r') cambia 'y_flat' en (sqrt(3)/2)*hex_size.
        # La altura de una "doble fila" (para cubrir un ciclo completo de patrón vertical) es sqrt(3)*hex_size.
        hex_row_pair_height = math.sqrt(3.0) * self.hex_size # Altura de dos filas de hexágonos apilados verticalmente
        
        # Altura total aproximada que se intenta cubrir con height_segments
        # Si height_segments = 1, queremos al menos una capa de hexágonos.
        # Si height_segments = N, queremos N "capas" de hexágonos.
        # La altura de una sola capa de hexágonos es aprox. sqrt(3)/2 * hex_size (si se mide por apotema)
        # o hex_size * sqrt(3) si se mide por la altura total del hexágono.
        # Usemos la altura de una fila de hexágonos (distancia vertical entre r y r+1)
        single_row_height_contribution = math.sqrt(3.0) * self.hex_size # de axial_to_cartesian_flat, y = ... + sqrt(3)*r*hex_size
        
        if self.height_segments > 0:
            # self.total_height_approx = self.height_segments * (math.sqrt(3.0) / 2.0 * self.hex_size) # Esto parece subestimar
            # Si cada segmento de altura corresponde a un incremento en 'r'
            self.total_height_approx = (self.height_segments) * single_row_height_contribution
        else: # Malla muy plana, solo una o dos "filas" de r
            self.total_height_approx = single_row_height_contribution * 1.5 # Suficiente para una capa

        # Los límites estrictos son el rango Z que queremos cubrir
        strict_min_z = -self.total_height_approx / 2.0
        strict_max_z = self.total_height_approx / 2.0
    
        # Un margen de exploración para el BFS para encontrar vecinos en los bordes Z
        # Debería ser al menos la máxima contribución Z de un vecino.
        height_margin = single_row_height_contribution * 1.5
    
        logger.info(f"Dims Calculadas: CircumActualSegs={self.circumference_segments_actual}, "
                    f"TotalAlturaAprox={self.total_height_approx:.2f}, "
                    f"RangoZEstricto=[{strict_min_z:.2f}, {strict_max_z:.2f}], MargenAltura={height_margin:.2f}")
    
        queue = deque()
        processed_coords = set() # Almacena tuplas (q, r)
    
        # Empezar desde (q=0, r=0)
        start_q, start_r = 0, 0
        queue.append((start_q, start_r))
        processed_coords.add((start_q, start_r))
        logger.debug(f"Celda inicial ({start_q},{start_r}) añadida a la cola.")
    
        cells_added_count = 0
        max_bfs_iterations = self.circumference_segments_actual * (self.height_segments + 4) * 10 # Límite heurístico
        if self.height_segments == 0: max_bfs_iterations = self.circumference_segments_actual * 20

        current_bfs_iteration = 0

        while queue and current_bfs_iteration < max_bfs_iterations:
            current_bfs_iteration += 1
            q, r = queue.popleft()
            axial_key = (q, r)
        
            # Convertir (q,r) axial a coordenadas cilíndricas
            # q_wrapped se usa para calcular x_flat para la periodicidad circunferencial
            q_for_x_flat = q % self.circumference_segments_actual
            # Si q es negativo, el módulo puede dar negativo en algunos Python, ajustar:
            if q_for_x_flat < 0: q_for_x_flat += self.circumference_segments_actual

            x_flat, y_flat = axial_to_cartesian_flat(q_for_x_flat, r, self.hex_size)
            # El 'q' original (no envuelto) se usa para el ángulo theta para evitar discontinuidades grandes
            # si la malla se extiende más allá de una vuelta en q antes de envolver.
            # No, para theta, debemos usar el q que define la posición en el cilindro desplegado.
            # La periodicidad la da el número de segmentos actual.
            # x_flat_for_theta = axial_to_cartesian_flat(q, r, self.hex_size)[0] # Usar q original para theta
            
            # Para theta, usamos el q original para que no salte, y luego normalizamos.
            # La x_flat para theta debe ser la del q original para que la espiral no se rompa.
            # El radio es self.radius.
            # theta = (q * 1.5 * self.hex_size) / self.radius -> esto es si q mapea directamente a x_flat
            # Mejor usar las funciones de conversión:
            x_flat_unwrapped, y_flat_unwrapped = axial_to_cartesian_flat(q, r, self.hex_size)
            cyl_radius_calc, cyl_theta_calc, cyl_z_calc = cartesian_flat_to_cylindrical(x_flat_unwrapped, y_flat_unwrapped, self.radius)

            # Determinar si la celda está dentro del rango Z estricto o solo dentro del margen de exploración
            is_within_strict_z = (strict_min_z - EPSILON <= cyl_z_calc <= strict_max_z + EPSILON)
            is_within_explore_margin_z = (strict_min_z - height_margin - EPSILON <= cyl_z_calc <= strict_max_z + height_margin + EPSILON)
        
            if not is_within_explore_margin_z:
                continue # Descartar y no explorar vecinos si está muy fuera de Z
        
            # Si está dentro del rango Z estricto, la añadimos a la malla
            # Usamos el (q,r) original como clave, la periodicidad se maneja al buscar vecinos.
            if is_within_strict_z:
                if axial_key not in self.cells:
                    # Usar q_axial original y r_axial original para la clave y los atributos de la celda
                    self.cells[axial_key] = Cell(self.radius, cyl_theta_calc, cyl_z_calc,
                                                 q, r) # q_vector se inicializa a ceros por defecto
                    cells_added_count += 1
                    if cells_added_count % 100 == 0:
                        logger.debug(f"--> Celda añadida #{cells_added_count}: {axial_key} (z={cyl_z_calc:.2f})")
            
            # Explorar vecinos si la celda está dentro del margen de exploración Z (incluso si no se añadió)
            theoretical_neighbors = self.get_axial_neighbors_coords(q, r) # Obtener solo coordenadas
            for nq, nr in theoretical_neighbors:
                neighbor_key = (nq, nr)
                if neighbor_key not in processed_coords:
                    # Verificar si el vecino potencial está dentro de un rango q razonable para evitar exploración infinita
                    # si la circunferencia no cierra bien y no hay periodicidad en q en la generación.
                    # El límite de q_exploration_limit es para el q original, no el envuelto.
                    q_exploration_limit_factor = 2 # Permitir explorar hasta 2 veces la circunferencia en q
                    if abs(nq) > self.circumference_segments_actual * q_exploration_limit_factor :
                        # logger.debug(f"Saltando vecino ({nq},{nr}) por exceder límite de exploración q.")
                        continue

                    # Calcular Z del vecino para decidir si añadirlo a la cola
                    _, neighbor_y_flat_unwrapped = axial_to_cartesian_flat(nq, nr, self.hex_size)
                    # _, _, neighbor_z_cyl = cartesian_flat_to_cylindrical(x_flat_nq_nr, y_flat_nq_nr, self.radius)

                    if strict_min_z - height_margin - EPSILON <= neighbor_y_flat_unwrapped <= strict_max_z + height_margin + EPSILON:
                        processed_coords.add(neighbor_key)
                        queue.append(neighbor_key)
    
        if current_bfs_iteration >= max_bfs_iterations:
             logger.warning(f"BFS detenido por alcanzar el límite de iteraciones ({max_bfs_iterations}). "
                            f"Celdas añadidas: {cells_added_count}. Puede ser normal para mallas grandes o indicar un problema.")
    
        logger.info(f"Malla inicializada con {len(self.cells)} celdas. ({current_bfs_iteration} iteraciones BFS)")

    def verify_connectivity(self, expected_min_neighbors_internal: int = 6) -> Dict[int, int]:
        """
        Verifica la conectividad interna de la malla contando los vecinos
        reales para cada celda. Loguea advertencias para celdas con conectividad inesperada.

        Args:
            expected_min_neighbors_internal (int): Número mínimo de vecinos esperado para celdas
                                                   que no están en un borde Z no periódico.

        Returns:
            Dict[int, int]: Distribución del número de vecinos por celda (NumVecinos: NumCeldas).
        """
        if not self.cells:
            logger.warning("Intento de verificar conectividad en una malla vacía.")
            return {}

        neighbor_counts = Counter()
        min_neighbors_found = 7 
        max_neighbors_found = -1
        cells_with_few_neighbors = 0

        logger.info(f"Verificando conectividad de {len(self.cells)} celdas (periodic_z={self.periodic_z})...")

        for cell_key, cell in self.cells.items():
            actual_neighbors = self.get_neighbor_cells(cell.q_axial, cell.r_axial)
            actual_neighbor_count = len(actual_neighbors)

            neighbor_counts[actual_neighbor_count] += 1
            min_neighbors_found = min(min_neighbors_found, actual_neighbor_count)
            max_neighbors_found = max(max_neighbors_found, actual_neighbor_count)

            is_on_z_border = False
            if not self.periodic_z and self.height_segments > 0: # Solo relevante si no es periódico en Z y tiene altura
                # Una celda está en el borde Z si alguno de sus vecinos teóricos en Z no existe
                # y no se encuentra un reemplazo periódico.
                # Simplificación: si su Z está cerca de min_z o max_z.
                if math.isclose(cell.z, self.min_z, abs_tol=self.hex_size*0.5) or \
                   math.isclose(cell.z, self.max_z, abs_tol=self.hex_size*0.5):
                    is_on_z_border = True
            
            expected_neighbors_for_this_cell = expected_min_neighbors_internal
            if is_on_z_border:
                # Celdas en bordes Z no periódicos pueden tener 3, 4 o 5 vecinos.
                # Es difícil dar un mínimo exacto sin analizar la topología local.
                # Pongamos un umbral más bajo para advertencia.
                if actual_neighbor_count < 3: 
                    logger.warning(f"  Celda en borde Z ({cell.q_axial},{cell.r_axial}, z={cell.z:.2f}) "
                                   f"tiene solo {actual_neighbor_count} vecinos.")
                    cells_with_few_neighbors +=1
            elif actual_neighbor_count < expected_min_neighbors_internal:
                logger.warning(f"  Celda interna ({cell.q_axial},{cell.r_axial}, z={cell.z:.2f}) "
                               f"tiene {actual_neighbor_count} vecinos (esperado >= {expected_neighbors_for_this_cell}).")
                cells_with_few_neighbors +=1
            
            if actual_neighbor_count > 6:
                 logger.error(f"  ERROR: Celda ({cell.q_axial},{cell.r_axial}) tiene {actual_neighbor_count} vecinos (>6). "
                              "Indica un error en la lógica de búsqueda de vecinos o duplicados.")

        result_dict = dict(sorted(neighbor_counts.items()))
        logger.info(f"Verificación de conectividad completada. Distribución (NumVecinos: NumCeldas): {result_dict}")
        if len(self.cells) > 0 : # Evitar división por cero si no hay celdas
            logger.info(f"  Mínimo vecinos: {min_neighbors_found}, Máximo vecinos: {max_neighbors_found}. "
                        f"Celdas con conectividad potencialmente baja: {cells_with_few_neighbors} ({cells_with_few_neighbors*100/len(self.cells):.1f}%).")
        else:
            logger.info(f"  Mínimo vecinos: N/A, Máximo vecinos: N/A (malla vacía).")

        if max_neighbors_found > 6 and len(self.cells)>0:
             logger.error("¡ERROR CRÍTICO DE CONECTIVIDAD! Se encontraron celdas con más de 6 vecinos.")
        
        return result_dict

    def get_cell(self, q: int, r: int) -> Optional[Cell]:
        """Obtiene una celda por sus coordenadas axiales (q, r) originales."""
        return self.cells.get((q, r))

    def get_axial_neighbors_coords(self, q: int, r: int) -> List[Tuple[int, int]]:
        """
        Obtiene las coordenadas axiales (q, r) de los 6 vecinos hexagonales teóricos
        de una celda dada por (q, r).
        """
        # Direcciones estándar para malla hexagonal en coordenadas axiales (q, r)
        # (también conocidas como "doubled coordinates" o "interlaced")
        # o "axial coordinates" donde q+r+s = 0 (s = -q-r)
        # axial_directions = [
        #     (+1,  0), (+1, -1), ( 0, -1),
        #     (-1,  0), (-1, +1), ( 0, +1)
        # ]
        # Para el sistema (q,r) donde q es columna y r es fila "sesgada":
        # Si r es par: (q+1,r), (q+1,r-1), (q,r-1), (q-1,r), (q,r+1), (q+1,r+1) ? No, esto es offset.
        # Usando el sistema axial (q,r) como en RedBlobGames:
        axial_directions = [
            (q + 1, r + 0), (q + 1, r - 1), (q + 0, r - 1),
            (q - 1, r + 0), (q - 1, r + 1), (q + 0, r + 1)
        ]
        return [(dq, dr) for dq, dr in axial_directions]

    def get_neighbor_cells(self, q_center: int, r_center: int) -> List[Cell]:
        """
        Obtiene los objetos Cell de los vecinos existentes de la celda (q_center, r_center).
        Maneja la periodicidad circunferencial (en q) y opcionalmente la periodicidad en Z.
        """
        neighbor_cells: List[Cell] = []
        # Celda central, para referencia de Z si es necesario (aunque no se usa aquí directamente)
        # center_cell = self.get_cell(q_center, r_center)
        # if not center_cell:
        #     logger.warning(f"get_neighbor_cells: Celda central ({q_center},{r_center}) no existe.")
        #     return []

        theoretical_neighbor_coords = self.get_axial_neighbors_coords(q_center, r_center)
    
        for nq_orig, nr_orig in theoretical_neighbor_coords:
            # 1. Aplicar periodicidad circunferencial a nq_orig para obtener nq_mapped
            #    La clave en self.cells usa (q_original, r_original)
            #    Pero para encontrar el vecino en una malla que "cierra", necesitamos mapear
            #    el q_original a un q_base dentro del rango [0, self.circumference_segments_actual - 1]
            #    y luego buscar si existe una celda cuyo q_original % circum == nq_mapped_base % circum
            #    y cuyo r sea nr_orig.
            #    Esto se complica si la malla se generó con q extendiéndose más allá de una vuelta.

            # Estrategia más simple:
            # Buscar directamente (nq_orig, nr_orig).
            # Si no existe, y la periodicidad está activa, buscar el equivalente envuelto.

            direct_neighbor = self.get_cell(nq_orig, nr_orig)
            if direct_neighbor:
                neighbor_cells.append(direct_neighbor)
                continue # Encontrado directamente, no buscar más para esta dirección

            # Si no se encontró directamente, considerar periodicidad
            # Periodicidad circunferencial (en q):
            # El q de una celda vecina puede estar "al otro lado" del cilindro.
            # El BFS ya debería haber generado celdas con q originales que pueden ser <0 o >circum_actual.
            # La clave (q,r) en self.cells es la coordenada "absoluta" del BFS.
            # El "cierre" del cilindro significa que una celda en q=circum_actual es vecina de q=0.
            
            # Para la periodicidad circunferencial:
            # Si nq_orig está fuera del rango [0, circum_actual-1] (o un rango similar si el origen no es 0),
            # necesitamos encontrar su equivalente dentro de ese rango.
            # Ejemplo: si circum_actual = 10. Vecino de q=0 es q=-1. Su equivalente es q=9.
            # Vecino de q=9 es q=10. Su equivalente es q=0.
            
            # El método _find_closest_z_neighbor ya maneja el mapeo de q.
            # Aquí, si no se encuentra (nq_orig, nr_orig), y self.periodic_z es True,
            # intentamos encontrar un vecino envuelto en Z.

            if self.periodic_z and self.total_height_approx > EPSILON:
                # Si no se encuentra directamente Y periodic_z está activo,
                # buscar un vecino envuelto verticalmente.
                # Necesitamos la coordenada Z teórica del vecino que no encontramos.
                # Y su coordenada q_original (nq_orig).
                
                # Calcular la Z teórica del vecino (nq_orig, nr_orig)
                x_flat_theoretical_neighbor, y_flat_theoretical_neighbor = axial_to_cartesian_flat(nq_orig, nr_orig, self.hex_size)
                _, _, z_theoretical_neighbor = cartesian_flat_to_cylindrical(x_flat_theoretical_neighbor, y_flat_theoretical_neighbor, self.radius)

                # El q que se pasa a _find_closest_z_neighbor es el q_original del vecino teórico
                wrapped_cell = self._find_closest_z_neighbor(nq_orig, z_theoretical_neighbor)
                if wrapped_cell:
                    # Asegurarse de no añadir un duplicado si de alguna manera ya estaba
                    if wrapped_cell not in neighbor_cells:
                         neighbor_cells.append(wrapped_cell)
                    # else:
                    #    logger.debug(f"Vecino envuelto ({wrapped_cell.q_axial},{wrapped_cell.r_axial}) ya estaba en la lista para ({q_center},{r_center}).")

        return neighbor_cells

    def _find_closest_z_neighbor(self, target_q_original: int, target_z_theoretical: float) -> Optional[Cell]:
        """
        Helper para encontrar una celda existente en la malla que sea la "versión envuelta en Z"
        de un vecino teórico (target_q_original, target_r_donde_sea_que_este_target_z).
        La celda encontrada debe tener una coordenada q_axial que, al ser envuelta circunferencialmente,
        coincida con target_q_original (envuelto), y su coordenada Z debe ser la más cercana
        a target_z_theoretical (considerando el envolvimiento en Z).

        Args:
            target_q_original (int): La coordenada q_axial original del vecino teórico que se busca.
            target_z_theoretical (float): La coordenada Z teórica del vecino que se busca.

        Returns:
            Optional[Cell]: La celda encontrada o None.
        """
        best_match: Optional[Cell] = None
        min_dz_abs_effective = float('inf')

        # El q_axial de las celdas en self.cells es el q "absoluto" del BFS.
        # El target_q_original también es "absoluto".
        # Para comparar 'q's, necesitamos envolver ambos al rango [0, circum_actual-1]
        target_q_wrapped_for_comparison = target_q_original % self.circumference_segments_actual
        if target_q_wrapped_for_comparison < 0:
            target_q_wrapped_for_comparison += self.circumference_segments_actual

        # logger.debug(f"    Buscando Z-vecino para q_orig={target_q_original} (envuelto={target_q_wrapped_for_comparison}), z_teor={target_z_theoretical:.2f}")

        if self.max_z - self.min_z <= EPSILON: # Altura real de la malla es cero o despreciable
            # logger.debug("    Altura real de la malla es cero, no se pueden encontrar vecinos envueltos en Z.")
            return None
        
        actual_mesh_height = self.max_z - self.min_z

        for cell_candidate in self.cells.values():
            # Comparar q: el q_axial del candidato envuelto debe coincidir con el target_q_original envuelto
            candidate_q_wrapped_for_comparison = cell_candidate.q_axial % self.circumference_segments_actual
            if candidate_q_wrapped_for_comparison < 0:
                candidate_q_wrapped_for_comparison += self.circumference_segments_actual
            
            if candidate_q_wrapped_for_comparison != target_q_wrapped_for_comparison:
                continue # No coincide en q (envuelto), no es el candidato correcto en la "columna"

            # Coincide en q (envuelto). Ahora comparar Z con envolvimiento.
            # Distancias a considerar para el envolvimiento en Z:
            # 1. Distancia directa a cell_candidate.z
            # 2. Distancia a cell_candidate.z + actual_mesh_height (como si target_z estuviera abajo)
            # 3. Distancia a cell_candidate.z - actual_mesh_height (como si target_z estuviera arriba)
            
            dz1 = abs(cell_candidate.z - target_z_theoretical)
            dz2 = abs(cell_candidate.z - (target_z_theoretical + actual_mesh_height))
            dz3 = abs(cell_candidate.z - (target_z_theoretical - actual_mesh_height))
            
            effective_dz = min(dz1, dz2, dz3)

            # Tolerancia para considerar una coincidencia: debe ser bastante cercano en Z.
            # Podría ser una fracción del tamaño del hexágono o de la altura de fila.
            z_match_tolerance = self.hex_size * 0.75 # Ajustar según sea necesario

            if effective_dz < min_dz_abs_effective and effective_dz < z_match_tolerance:
                min_dz_abs_effective = effective_dz
                best_match = cell_candidate
                # logger.debug(f"      -> Nuevo mejor match Z-vecino: ({best_match.q_axial},{best_match.r_axial}) z={best_match.z:.2f} con dz_eff={min_dz_abs_effective:.4f}")
        
        # if best_match:
        #     logger.debug(f"    Encontrado Z-vecino: ({best_match.q_axial},{best_match.r_axial}) para q_orig={target_q_original}, z_teor={target_z_theoretical:.2f}")
        # else:
        #     logger.debug(f"    No se encontró Z-vecino para q_orig={target_q_original}, z_teor={target_z_theoretical:.2f} (min_dz_eff={min_dz_abs_effective:.2f} > tol={z_match_tolerance:.2f})")
            
        return best_match

    def get_all_cells(self) -> List[Cell]:
        """Retorna una lista de todas las celdas en la malla."""
        return list(self.cells.values())

    def to_dict(self) -> Dict[str, Any]:
        """Retorna una representación de la malla completa como diccionario."""
        return {
            "metadata": {
                "radius": self.radius,
                "height_segments": self.height_segments,
                "circumference_segments_actual": self.circumference_segments_actual,
                "hex_size": self.hex_size,
                "periodic_z": self.periodic_z,
                "num_cells": len(self.cells),
                "z_bounds": {"min": self.min_z, "max": self.max_z},
                "total_height_approx": self.total_height_approx,
                "previous_flux": self.previous_flux # Incluir para estado completo si es necesario
            },
            "cells": [cell.to_dict() for cell in self.cells.values()]
        }