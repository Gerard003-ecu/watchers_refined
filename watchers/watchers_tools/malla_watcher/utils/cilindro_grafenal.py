# watchers/watchers_tools/malla_watcher/utils/cilindro_grafenal.py

import math
import logging
import numpy as np
from collections import deque, Counter
from typing import List, Optional, Dict, Tuple, Any

# --- Configuración del Logging ---
logger = logging.getLogger(__name__)  # Usar __name__ para logger

# Ejemplo de configuración básica
# si se ejecuta directamente o para pruebas:
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter(
#         "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
#     )
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

EPSILON = 1e-9


def axial_to_cartesian_flat(
        q: int, r: int, hex_size: float = 1.0
) -> tuple[float, float]:
    """Convierte coordenadas axiales (q, r)
    a cartesianas (x, y) en malla hexagonal plana.
    """
    x = hex_size * (3. / 2 * q)
    y = hex_size * (math.sqrt(3) / 2 * q + math.sqrt(3) * r)
    return x, y


def cartesian_flat_to_cylindrical(
        x_flat: float, y_flat: float, radius: float
) -> tuple[float, float, float]:
    """
    Enrolla coordenadas cartesianas planas (x, y)
    en un cilindro de radio 'radius'.
    El eje 'x' plano se convierte en dirección azimutal (theta).
    El eje 'y' plano se convierte en dirección axial (z).
    """
    theta = x_flat / radius if radius > EPSILON else 0.0
    z = y_flat
    theta = theta % (2 * math.pi)
    if theta < 0:
        theta += (2 * math.pi)
    return radius, theta, z


class Cell:
    """
    Representa una celda (nodo) de la malla hexagonal cilíndrica.
    Almacena coordenadas cilíndricas (r, theta, z), estado local de oscilador
    (amplitud, velocidad) y valor del campo vectorial externo (q_vector).
    """
    def __init__(
            self,
            cyl_radius: float,
            cyl_theta: float,
            cyl_z: float,
            q_axial: int,
            r_axial: int,
            amplitude: float = 0.0,
            velocity: float = 0.0,
            q_vector: Optional[np.ndarray] = None
    ):
        """
        Inicializa la celda.

        Args:
            cyl_radius (float): Radio cilíndrico.
            cyl_theta (float): Ángulo azimutal (radianes, [0, 2*pi)).
            cyl_z (float): Altura a lo largo del eje.
            q_axial (int): Coordenada axial 'q' original.
            r_axial (int): Coordenada axial 'r' original.
            amplitude (float): Amplitud del oscilador.
            velocity (float): Velocidad del oscilador.
            q_vector (Optional[np.ndarray]): Campo vectorial externo [vx, vy].
        """
        self.r: float = cyl_radius
        self.theta: float = cyl_theta
        self.z: float = cyl_z
        self.q_axial: int = q_axial
        self.r_axial: int = r_axial
        self.amplitude: float = amplitude
        self.velocity: float = velocity
        # ATRIBUTO VORONOI CONSERVADO
        self.voronoi_neighbors: List[Cell] = []

        if q_vector is not None and isinstance(q_vector, np.ndarray):
            self.q_vector: np.ndarray = q_vector
        else:
            self.q_vector: np.ndarray = np.zeros(2, dtype=float)

    def __repr__(self) -> str:
        q_vec_str = f"[{self.q_vector[0]:.2f}, {self.q_vector[1]:.2f}]"
        return (
            f"Cell(ax=({self.q_axial},{self.r_axial}), "
            f"cyl=(r={self.r:.2f}, θ={self.theta:.2f}, z={self.z:.2f}), "
            f"amp={self.amplitude:.2f}, vel={self.velocity:.2f}, "
            f"q_v={q_vec_str})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Retorna una representación de la celda como diccionario.
        """
        return {
            "axial_coords": {"q": self.q_axial, "r": self.r_axial},
            "cylindrical_coords":
                {"r": self.r, "theta": self.theta, "z": self.z},
            "amplitude": self.amplitude,
            "velocity": self.velocity,
            "q_vector": self.q_vector.tolist(),
            "voronoi_neighbors_count": len(self.voronoi_neighbors),
        }


class HexCylindricalMesh:
    """
    Gestiona malla hexagonal cilíndrica, incluyendo creación,
    validación de conectividad y acceso a celdas.
    """
    def __init__(
            self,
            radius: float,
            height_segments: int,
            circumference_segments_target: int,
            hex_size: float = 1.0,
            periodic_z: bool = False
    ):
        """
        Inicializa la malla hexagonal cilíndrica.

        Args:
            radius (float): Radio del cilindro.
            height_segments (int): Número de segmentos en altura.
            circumference_segments_target (int): Segmentos para cerrar la
                                               circunferencia.
            hex_size (float): Tamaño característico de los hexágonos.
            periodic_z (bool): Condiciones periódicas en dirección Z.

        Raises:
            ValueError: Si parámetros son inválidos.
        """
        if radius <= 0:
            raise ValueError("El radio del cilindro debe ser positivo.")
        if hex_size <= 0:
            raise ValueError("El tamaño del hexágono debe ser positivo.")
        if radius < hex_size:
            logger.warning(
                f"El radio ({radius}) es menor que el tamaño del hexágono "
                f"({hex_size}). "
                "La malla podría ser degenerada."
            )
        if height_segments < 0:
            raise ValueError(
                "El número de segmentos de altura no puede ser negativo."
            )
        if circumference_segments_target < 3:
            raise ValueError(
                "Se requieren al menos 3 segmentos de circunferencia objetivo."
            )

        self.radius: float = radius
        self.height_segments: int = max(0, height_segments)
        self.hex_size: float = hex_size
        self.periodic_z: bool = periodic_z

        self.circumference: float = 2 * math.pi * self.radius
        hex_width_circumferential = self.hex_size * 1.5

        if hex_width_circumferential < EPSILON:
            raise ValueError(
                "El tamaño del hexágono es demasiado pequeño, "
                "resultando en ancho circunferencial nulo."
            )

        self.circumference_segments_actual = max(
            3, circumference_segments_target
        )
        # Fix for original Line 126 E501
        actual_circ_covered = (
            self.circumference_segments_actual * hex_width_circumferential
        )
        self.actual_circumference_covered_by_q_segments = actual_circ_covered

        logger.info(
            f"Malla Cilíndrica: Radio={self.radius:.2f}, "
            f"AlturaSeg={self.height_segments}, "
            f"CircumSegTarget={circumference_segments_target} -> "
            f"Actual={self.circumference_segments_actual}, "
            f"HexSize={self.hex_size:.2f}, "
            f"PeriodicZ={self.periodic_z}"
        )
        if (not math.isclose(
            self.actual_circumference_covered_by_q_segments,  # Line 197
            self.circumference,
            rel_tol=0.15
        ) and self.circumference_segments_actual > 3):
            logger.warning(
                f"La circunferencia teórica ({self.circumference:.2f}) y la "
                f"cubierta por segmentos q "  # Line 200
                f"({self.actual_circumference_covered_by_q_segments:.2f}) "
                f"difieren mucho."
            )

        self.cells: Dict[Tuple[int, int], Cell] = {}
        self.min_z: float = 0.0
        self.max_z: float = 0.0
        self.total_height_approx: float = 0.0
        self.previous_flux: float = 0.0

        self._initialize_mesh()
        if self.cells:
            self._calculate_z_bounds()
            self.verify_connectivity()
        else:
            logger.error("¡La inicialización de la malla no generó celdas!")

    def _calculate_z_bounds(self):
        """Calcula y actualiza los límites Z (mínimo y máximo) de la malla.

        Este método itera sobre todas las celdas existentes en la malla para
        encontrar los valores mínimo y máximo de sus coordenadas 'z'. Los
        resultados se almacenan en los atributos `self.min_z` y `self.max_z`.
        Se llama típicamente después de que la malla ha sido inicializada.
        Si la malla está vacía, `min_z` y `max_z` se establecen en 0.0.
        """
        if not self.cells:
            self.min_z = 0.0
            self.max_z = 0.0
            logger.warning("Intentando calcular límites Z en malla vacía.")
            return

        z_values = [cell.z for cell in self.cells.values()]
        self.min_z = min(z_values)
        self.max_z = max(z_values)
        logger.info(
            f"Límite Z calculado: min={self.min_z:.2f}, max={self.max_z:.2f}."
            f"Altura real: {self.max_z - self.min_z:.2f}. "
            f"Altura aprox.: {self.total_height_approx:.2f}"
        )

    def _initialize_mesh(self):
        """Construye las celdas de la malla utilizando un enfoque BFS.

        Este es el método principal para generar la geometría de la malla.
        Comienza desde una celda de origen (0,0) en coordenadas axiales y
        explora hacia afuera utilizando una búsqueda en amplitud (BFS) para
        agregar nuevas celdas.

        Las celdas se generan en un plano 2D usando coordenadas axiales y luego
        se mapean a la superficie cilíndrica. Se aplican filtros basados en la
        altura (coordenada Z) para asegurar que la malla se ajuste a las
        dimensiones deseadas (`height_segments`). La expansión circunferencial
        está limitada por `circumference_segments_actual`.

        El proceso BFS tiene un límite de iteraciones para prevenir bucles
        infinitos en configuraciones problemáticas. Las celdas creadas se
        almacenan en el diccionario `self.cells`.
        """
        logger.info(
            f"Inicializando malla hexagonal cilíndrica "
            f"(periodic_z={self.periodic_z})..."
        )
        self.cells.clear()

        single_row_height_contribution = math.sqrt(3.0) * self.hex_size

        if self.height_segments > 0:
            self.total_height_approx = (
                self.height_segments * single_row_height_contribution
            )
        else:
            self.total_height_approx = single_row_height_contribution * 1.5

        strict_min_z = -self.total_height_approx / 2.0
        strict_max_z = self.total_height_approx / 2.0
        height_margin = single_row_height_contribution * 1.5

        logger.info(
            f"Dims Calculadas: CircumActualSegs="
            f"{self.circumference_segments_actual}, "
            f"TotalAlturaAprox={self.total_height_approx:.2f}, "
            f"RangoZEstricto=[{strict_min_z:.2f}, {strict_max_z:.2f}], "
            f"MargenAltura={height_margin:.2f}"
        )

        queue = deque()
        processed_coords = set()

        start_q, start_r = 0, 0
        queue.append((start_q, start_r))
        processed_coords.add((start_q, start_r))
        logger.debug(f"Celda inicial ({start_q},{start_r}) añadida.")

        cells_added_count = 0
        # Fix for original Line 577 E502 (and E501)
        max_bfs_iter_calc = (
            self.circumference_segments_actual *
            (self.height_segments + 4) * 10
        )
        max_bfs_iterations = max_bfs_iter_calc
        if self.height_segments == 0:
            max_bfs_iterations = (
                self.circumference_segments_actual * 20
            )

        current_bfs_iteration = 0

        while queue and current_bfs_iteration < max_bfs_iterations:
            current_bfs_iteration += 1
            q, r = queue.popleft()
            axial_key = (q, r)

            q_for_x_flat = q % self.circumference_segments_actual
            if q_for_x_flat < 0:
                q_for_x_flat += self.circumference_segments_actual

            x_flat_unwrapped, y_flat_unwrapped = axial_to_cartesian_flat(
                q, r, self.hex_size
            )
            _, cyl_theta_calc, cyl_z_calc = cartesian_flat_to_cylindrical(
                x_flat_unwrapped, y_flat_unwrapped, self.radius
            )

            is_within_strict_z = (
                strict_min_z - EPSILON <= cyl_z_calc <= strict_max_z + EPSILON
            )
            is_within_explore_margin_z = (
                (strict_min_z - height_margin - EPSILON <= cyl_z_calc) and
                (cyl_z_calc <= strict_max_z + height_margin + EPSILON)
            )

            if not is_within_explore_margin_z:
                continue

            if is_within_strict_z:
                if axial_key not in self.cells:
                    self.cells[axial_key] = Cell(
                        self.radius, cyl_theta_calc, cyl_z_calc,
                        q, r
                    )
                    cells_added_count += 1
                    if cells_added_count % 100 == 0:
                        logger.debug(
                            f"--> Celda añadida #{cells_added_count}: "
                            f"{axial_key} (z={cyl_z_calc:.2f})"
                        )

            theoretical_neighbors = self.get_axial_neighbors_coords(q, r)
            for nq, nr in theoretical_neighbors:
                neighbor_key = (nq, nr)
                if neighbor_key not in processed_coords:
                    q_exploration_limit_factor = 2
                    q_exp_limit = (
                        self.circumference_segments_actual *
                        q_exploration_limit_factor
                    )
                    # Condición dividida para cumplir PEP8
                    if (abs(nq) >
                            q_exp_limit):
                        continue

                    _, neighbor_y_flat_unwrapped = axial_to_cartesian_flat(
                        nq, nr, self.hex_size
                    )

                    # Line 276
                    z_lower_bound = strict_min_z - height_margin - EPSILON
                    z_upper_bound = strict_max_z + height_margin + EPSILON
                    if (z_lower_bound <= neighbor_y_flat_unwrapped <=
                            z_upper_bound):
                        processed_coords.add(neighbor_key)
                        queue.append(neighbor_key)

        if current_bfs_iteration >= max_bfs_iterations:
            logger.warning(
                f"BFS detenido por límite ({max_bfs_iterations}). "
                f"Celdas añadidas: {cells_added_count}."
            )

        logger.info(
            f"Malla inicializada con {len(self.cells)} celdas. "
            f"({current_bfs_iteration} iteraciones BFS)"
        )

    def verify_connectivity(
            self,
            expected_min_neighbors_internal: int = 6
    ) -> Dict[int, int]:
        """Verifica la conectividad de las celdas en la malla.

        Este método itera sobre todas las celdas de la malla y cuenta cuántos
        vecinos reales tiene cada una (utilizando `get_neighbor_cells`).
        Compara este número con un mínimo esperado, especialmente para celdas
        internas. Registra advertencias para celdas con conectividad
        inesperadamente baja y errores si alguna celda tiene más de 6 vecinos
        (lo cual indicaría un problema en la lógica de la malla hexagonal).

        Args:
            expected_min_neighbors_internal: El número mínimo de vecinos que
                se espera para una celda interna (no en el borde Z, si la malla
                no es periódica en Z). Por defecto es 6.

        Returns:
            Diccionario que representa la distribución del número de vecinos.
            Las claves son el número de vecinos contados, y los valores son
            cuántas celdas tienen ese número de vecinos. Retorna un diccionario
            vacío si la malla no tiene celdas.
        """
        if not self.cells:
            logger.warning("Intento de verificar conectividad en malla vacía.")
            return {}

        neighbor_counts = Counter()
        min_neighbors_found = 7
        max_neighbors_found = -1
        cells_with_few_neighbors = 0

        logger.info(
            f"Verificando conectividad de {len(self.cells)} celdas "
            f"(periodic_z={self.periodic_z})..."
        )

        for cell in self.cells.values():
            actual_neighbors = self.get_neighbor_cells(
                cell.q_axial, cell.r_axial
            )
            actual_neighbor_count = len(actual_neighbors)

            neighbor_counts[actual_neighbor_count] += 1
            min_neighbors_found = min(
                min_neighbors_found, actual_neighbor_count
            )
            max_neighbors_found = max(
                max_neighbors_found, actual_neighbor_count
            )

            is_on_z_border = False
            if not self.periodic_z and self.height_segments > 0:
                # Line 328
                is_close_min_z = math.isclose(
                    cell.z, self.min_z, abs_tol=self.hex_size * 0.5
                )
                is_close_max_z = math.isclose(
                    cell.z, self.max_z, abs_tol=self.hex_size * 0.5
                )
                if is_close_min_z or is_close_max_z:
                    is_on_z_border = True

            if is_on_z_border:
                if actual_neighbor_count < 3:
                    logger.warning(
                        f"  Celda en borde Z ({cell.q_axial},{cell.r_axial}, "
                        f"z={cell.z:.2f}) tiene solo "
                        f"{actual_neighbor_count} vecinos."
                    )
                    cells_with_few_neighbors += 1
            elif actual_neighbor_count < expected_min_neighbors_internal:
                logger.warning(  # Line 335
                    f"  Celda interna ({cell.q_axial},{cell.r_axial}, "
                    f"z={cell.z:.2f}) tiene {actual_neighbor_count} "
                    "vecinos (esperado >= "
                    f"{expected_min_neighbors_internal})."
                )
                cells_with_few_neighbors += 1

            if actual_neighbor_count > 6:
                logger.error(
                    f"  ERROR: Celda ({cell.q_axial},{cell.r_axial}) tiene "
                    f"{actual_neighbor_count} vecinos (>6). Indica error."
                )

        result_dict = dict(sorted(neighbor_counts.items()))
        logger.info(
            "Verificación completada. "
            f"Distribución: {result_dict}"
        )
        if self.cells:
            percentage_low_connectivity = (
                cells_with_few_neighbors * 100 / len(self.cells)
            )
            # Fix for original Line 382 E501
            logger.info(
                f"  Mínimo vecinos: {min_neighbors_found}, "
                f"Máximo: {max_neighbors_found}. "
                f"Celdas con baja conectividad: {cells_with_few_neighbors} "
                f"({percentage_low_connectivity:.1f}%)."
            )
        else:
            log_message = "  Mínimo vecinos: N/A, Máximo vecinos: N/A."
            logger.info(log_message)

        if max_neighbors_found > 6 and self.cells:
            logger.error(
                "¡ERROR CRÍTICO DE CONECTIVIDAD! "
                "Celdas con más de 6 vecinos."
            )

        return result_dict

    def get_cell(self, q: int, r: int) -> Optional[Cell]:
        """Recupera una celda de la malla por sus coordenadas axiales.

        Args:
            q: La coordenada axial 'q' de la celda a buscar.
            r: La coordenada axial 'r' de la celda a buscar.

        Returns:
            La instancia de `Cell` correspondiente a las coordenadas (q, r)
            si existe en la malla; de lo contrario, retorna `None`.
        """
        return self.cells.get((q, r))

    def get_axial_neighbors_coords(
            self, q: int, r: int
    ) -> List[
        Tuple[int, int]
    ]:
        """Calcula las coordenadas axiales de 6 vecinos teóricos de una celda.

        Dada una celda central definida por sus coordenadas axiales (q, r),
        esta función retorna una lista de las coordenadas axiales (q, r) de
        sus seis celdas vecinas directas en la malla hexagonal ideal.
        No verifica si las celdas vecinas existen realmente en la malla actual.

        Args:
            q: La coordenada axial 'q' de la celda central.
            r: La coordenada axial 'r' de la celda central.

        Returns:
            Una lista de tuplas, donde cada tupla contiene las coordenadas
            (q, r) de un vecino teórico.
        """
        axial_directions = [
            (q + 1, r + 0),  # Derecha
            (q + 1, r - 1),  # Arriba-derecha
            (q + 0, r - 1),  # Arriba-izquierda
            (q - 1, r + 0),  # Izquierda
            (q - 1, r + 1),  # Abajo-izquierda
            (q + 0, r + 1),  # Abajo-derecha
        ]
        return [(dq, dr) for dq, dr in axial_directions]

    def get_neighbor_cells(
            self, q_center: int, r_center: int
    ) -> List[Cell]:
        """Obtiene las celdas vecinas existentes de una celda central.

        Calcula los vecinos teóricos de la celda en (q_center, r_center) y
        luego verifica cuáles de estos vecinos existen realmente en la malla.
        Maneja la periodicidad circunferencial (alrededor del eje del cilindro)
        automáticamente. Si `self.periodic_z` es True, también intenta
        encontrar vecinos envueltos en la dirección Z (altura).

        Args:
            q_center: La coordenada axial 'q' de la celda central.
            r_center: La coordenada axial 'r' de la celda central.

        Returns:
            Una lista de instancias de `Cell` que son vecinas directas y
            existentes de la celda central. La lista puede estar vacía si
            no se encuentran vecinos o si la celda central no existe.
        """
        neighbor_cells: List[Cell] = []
        theoretical_neighbor_coords = self.get_axial_neighbors_coords(
            q_center, r_center
        )

        for nq_orig, nr_orig in theoretical_neighbor_coords:
            direct_neighbor = self.get_cell(nq_orig, nr_orig)
            if direct_neighbor:
                neighbor_cells.append(direct_neighbor)
                continue

            if self.periodic_z and self.total_height_approx > EPSILON:
                # Line 394
                x_flat_theoretical, y_flat_theoretical = \
                    axial_to_cartesian_flat(
                        nq_orig, nr_orig, self.hex_size
                    )
                # Line 395
                _, _, z_theoretical = cartesian_flat_to_cylindrical(
                    x_flat_theoretical,
                    y_flat_theoretical,
                    self.radius
                )

                wrapped_cell = self._find_closest_z_neighbor(
                    nq_orig, z_theoretical
                )
                if wrapped_cell and wrapped_cell not in neighbor_cells:
                    neighbor_cells.append(wrapped_cell)

        return neighbor_cells

    def _find_closest_z_neighbor(  # Line 402
            self,
            target_q_original: int,
            target_z_theoretical: float
    ) -> Optional[Cell]:
        """Busca una celda existente que corresponda a un vecino envuelto en Z.

        Este método se utiliza cuando `periodic_z` es True. Dada una coordenada
        'q' axial original de un vecino teórico y su coordenada 'z' teórica
        (que podría estar fuera de los límites `min_z`, `max_z` de la malla),
        intenta encontrar una celda existente en la malla que tenga la misma
        coordenada 'q' (o su equivalente envuelto circunferencialmente) y una
        coordenada 'z' que, al considerar la periodicidad en altura, sea la más
        cercana a `target_z_theoretical`.

        Args:
            target_q_original: La coordenada axial 'q' original del vecino
                teórico que se está buscando.
            target_z_theoretical: La coordenada 'z' teórica (altura) del
                vecino que se está buscando. Esta 'z' podría estar fuera de los
                límites actuales de la malla si el vecino es una envoltura
                periódica.

        Returns:
            La instancia de `Cell` que mejor coincide como un vecino envuelto
            en Z, o `None` si no se encuentra ninguna coincidencia adecuada
            dentro de una tolerancia.
        """
        best_match: Optional[Cell] = None
        min_dz_abs_effective = float('inf')

        target_q_wrapped = (
            target_q_original % self.circumference_segments_actual
        )
        if target_q_wrapped < 0:
            target_q_wrapped += self.circumference_segments_actual

        if self.max_z - self.min_z <= EPSILON:
            return None

        actual_mesh_height = self.max_z - self.min_z

        for cell_candidate in self.cells.values():
            candidate_q_wrapped = (
                cell_candidate.q_axial % self.circumference_segments_actual
            )
            if candidate_q_wrapped < 0:
                candidate_q_wrapped += self.circumference_segments_actual

            if candidate_q_wrapped != target_q_wrapped:
                continue

            dz1 = abs(cell_candidate.z - target_z_theoretical)
            dz2 = abs(cell_candidate.z -
                      (target_z_theoretical + actual_mesh_height))
            dz3 = abs(cell_candidate.z -
                      (target_z_theoretical - actual_mesh_height))

            effective_dz = min(dz1, dz2, dz3)
            z_match_tolerance = self.hex_size * 0.75

            # Line 452
            if (effective_dz < min_dz_abs_effective and
                    effective_dz < z_match_tolerance):
                min_dz_abs_effective = effective_dz
                best_match = cell_candidate

        return best_match

    def get_all_cells(self) -> List[Cell]:
        """Retorna una lista con todas las instancias de `Cell` en la malla.

        Returns:
            Una lista de objetos `Cell`. Si la malla está vacía, retorna una
            lista vacía.
        """
        return list(self.cells.values())

    def compute_voronoi_neighbors(self, periodic_theta: bool = True) -> None:
        """Calcula los vecinos de Voronoi para cada celda de la malla.

        Utiliza `scipy.spatial.Voronoi` para calcular la teselación de Voronoi
        basada en las coordenadas (theta, z) de las celdas. Luego, para cada
        celda, identifica sus celdas vecinas en el diagrama de Voronoi.
        Este método actualiza el atributo `voronoi_neighbors` de cada `Cell`
        en la malla.

        La periodicidad en la dirección theta (circunferencial) se maneja
        calculando el diagrama en el espacio desenrollado y luego buscando
        vecinos periódicos para las celdas en los bordes.

        Requiere que `scipy` esté instalado. Si `scipy.spatial.Voronoi` no
        puede ser importado, se registra un error y el método no hace nada.

        Args:
            periodic_theta: Si es True (por defecto), se considera la
                periodicidad en la dirección theta al calcular los vecinos.
                Si es False, el cálculo se realiza como si la malla fuera una
                franja plana desenrollada.
        """
        try:
            from scipy.spatial import Voronoi, cKDTree
        except ImportError:
            logger.error(
                "Scipy no instalado. Voronoi no disp."
            )
            return

        if not self.cells:
            logger.warning(
                "Malla vacía. No se calculan vecinos Voronoi."
            )
            return

        # Recopilar puntos (theta, z) y crear lista de celdas indexadas
        points: List[Tuple[float, float]] = []
        original_cells: List[Cell] = []
        for cell in self.cells.values():
            points.append((cell.theta, cell.z))
            original_cells.append(cell)

        n_original = len(points)
        if n_original < 3:
            logger.warning(
                "Muy pocas celdas para Voronoi."
            )
            return

        points_array = np.array(points)
        vor = Voronoi(points_array)

        # Construir diccionario de vecinos
        neighbor_dict = {i: set() for i in range(n_original)}
        for ridge in vor.ridge_points:
            i, j = ridge
            neighbor_dict[i].add(j)
            neighbor_dict[j].add(i)

        for i in range(n_original):
            cell = original_cells[i]
            voronoi_neighbors_indices = neighbor_dict[i]
            cell.voronoi_neighbors = [original_cells[j] for j in voronoi_neighbors_indices]

        if periodic_theta:
            tree = cKDTree(points_array)
            border_tol = self.hex_size / self.radius if self.radius > 0 else 0.1

            left_border_indices = [i for i, p in enumerate(points) if p[0] < border_tol]
            for i in left_border_indices:
                p = points_array[i]
                p_shifted = (p[0] + 2 * math.pi, p[1])
                dist, j = tree.query(p_shifted, k=1)
                if dist < border_tol * 2:
                    if original_cells[j] not in original_cells[i].voronoi_neighbors:
                        original_cells[i].voronoi_neighbors.append(original_cells[j])
                    if original_cells[i] not in original_cells[j].voronoi_neighbors:
                        original_cells[j].voronoi_neighbors.append(original_cells[i])

        logger.info(
            f"Vecinos Voronoi calculados para {n_original} celdas."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la malla completa a una representación de diccionario.

        Este método serializa los metadatos de la malla (como radio, número de
        segmentos, etc.) y una lista de todas las celdas convertidas también
        a diccionarios (usando `Cell.to_dict()`).

        Returns:
            Un diccionario que contiene:
            - 'metadata': Un diccionario con los parámetros de configuración y
                          propiedades calculadas de la malla.
            - 'cells': Una lista de diccionarios, donde cada diccionario es la
                       representación de una celda en la malla.
        """
        return {
            "metadata": {
                "radius": self.radius,
                "height_segments": self.height_segments,
                "circ_segments_actual":
                    self.circumference_segments_actual,
                "hex_size": self.hex_size,
                "periodic_z": self.periodic_z,
                "num_cells": len(self.cells),
                "z_bounds": {
                    "min": self.min_z,
                    "max": self.max_z
                },
                "total_height_approx": self.total_height_approx,
                "previous_flux": self.previous_flux
                # fluxo do passo anterior
            },
            "cells": [cell.to_dict() for cell in self.cells.values()]
        }
