# watchers/watchers_tools/malla_watcher/utils/cilindro_grafenal.py

import math
import logging
import numpy as np
from collections import deque, Counter
from typing import List, Optional, Dict, Tuple, Any, Set

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

        if (q_vector is not None and
                isinstance(q_vector, np.ndarray) and
                q_vector.shape == (2,)):
            self.q_vector: np.ndarray = q_vector
        else:
            self.q_vector: np.ndarray = np.zeros(2, dtype=float)

        # Nuevo atributo para vecinos de Voronoi
        self.voronoi_neighbors: List[Cell] = []

    def __repr__(self) -> str:
        q_vec_str = f"[{self.q_vector[0]:.2f}, {self.q_vector[1]:.2f}]"
        return (
            f"Cell(ax=({self.q_axial},{self.r_axial}), "
            f"cyl=(r={self.r:.2f}, θ={self.theta:.2f}, z={self.z:.2f}), "
            f"amp={self.amplitude:.2f}, vel={self.velocity:.2f}, "
            f"q_v={q_vec_str})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Retorna representación serializable a JSON."""
        return {
            "axial_coords": {"q": self.q_axial, "r": self.r_axial},
            "cylindrical_coords": {"r": self.r, "theta": self.theta, "z": self.z},
            "amplitude": self.amplitude,
            "velocity": self.velocity,
            "q_vector": self.q_vector.tolist()
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
            circumference_segments_target: int,  # Line 116
            hex_size: float = 1.0,
            periodic_z: bool = False
    ):
        """
        Inicializa la malla hexagonal cilíndrica.

        Args:
            radius (float): Radio del cilindro.
            height_segments (int): Número de segmentos en altura.
            circumference_segments_target (int): Número deseado para cerrar
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
                f"({hex_size}). La malla podría ser degenerada."
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
        self.actual_circumference_covered_by_q_segments = (  # Line 142
            self.circumference_segments_actual *
            hex_width_circumferential
        )

        logger.info(
            f"Malla Cilíndrica: Radio={self.radius:.2f}, "
            f"AlturaSeg={self.height_segments}, "
            f"CircumSegTarget={circumference_segments_target} -> "  # Line 196
            f"Actual={self.circumference_segments_actual}, "
            f"HexSize={self.hex_size:.2f}, PeriodicZ={self.periodic_z}"
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
                f"difieren significativamente."
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
        """Calcula límites Z mínimo y máximo reales."""
        if not self.cells:
            self.min_z = 0.0
            self.max_z = 0.0
            logger.warning("Intentando calcular límites Z en malla vacía.")
            return

        z_values = [cell.z for cell in self.cells.values()]
        self.min_z = min(z_values)
        self.max_z = max(z_values)
        logger.info(
            f"Límites Z calculados: min={self.min_z:.2f}, max={self.max_z:.2f}. "
            f"Altura real: {self.max_z - self.min_z:.2f}. "
            f"Altura aprox.: {self.total_height_approx:.2f}"
        )

    def _initialize_mesh(self):
        """Crea celdas usando BFS con coordenadas axiales (q,r)."""
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
        max_bfs_iterations = (  # Line 232
            self.circumference_segments_actual *
            (self.height_segments + 4) * 10
        )
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
                    if abs(nq) > q_exp_limit:
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
        """
        Verifica conectividad interna contando vecinos reales.

        Args:
            expected_min_neighbors_internal: Mínimo esperado para celdas internas.

        Returns:
            Dict[int, int]: Distribución de vecinos por celda.
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
                        f"z={cell.z:.2f}) tiene solo {actual_neighbor_count} vecinos."
                    )
                    cells_with_few_neighbors += 1
            elif actual_neighbor_count < expected_min_neighbors_internal:
                logger.warning(  # Line 335
                    f"  Celda interna ({cell.q_axial},{cell.r_axial}, "
                    f"z={cell.z:.2f}) tiene {actual_neighbor_count} "
                    f"vecinos (esperado >= {expected_min_neighbors_internal})."
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
            logger.info(
                f"  Mínimo vecinos: {min_neighbors_found}, "
                f"Máximo: {max_neighbors_found}. Celdas con baja "  # Line 359
                f"conectividad: {cells_with_few_neighbors} "
                f"({percentage_low_connectivity:.1f}%)."
            )
        else:
            logger.info("  Mínimo vecinos: N/A, Máximo vecinos: N/A.")

        if max_neighbors_found > 6 and self.cells:
            logger.error(
                "¡ERROR CRÍTICO DE CONECTIVIDAD! "
                "Celdas con más de 6 vecinos."
            )

        return result_dict

    def get_cell(self, q: int, r: int) -> Optional[Cell]:
        """Obtiene celda por coordenadas axiales (q, r)."""
        return self.cells.get((q, r))

    def get_axial_neighbors_coords(
            self, q: int, r: int
    ) -> List[Tuple[int, int]]:
        """Obtiene coordenadas axiales de los 6 vecinos hexagonales teóricos."""
        axial_directions = [
            (q + 1, r + 0), (q + 1, r - 1), (q + 0, r - 1),
            (q - 1, r + 0), (q - 1, r + 1), (q + 0, r + 1)
        ]
        return [(dq, dr) for dq, dr in axial_directions]

    def get_neighbor_cells(
            self, q_center: int, r_center: int
    ) -> List[Cell]:
        """
        Obtiene objetos Cell de vecinos existentes.
        Maneja periodicidad circunferencial y opcionalmente en Z.
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
                    x_flat_theoretical, y_flat_theoretical,
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
        """
        Encuentra celda existente que sea "versión envuelta en Z".

        Args:
            target_q_original: q_axial del vecino teórico.
            target_z_theoretical: Z teórica del vecino.

        Returns:
            Optional[Cell]: Celda encontrada o None.
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
            dz2 = abs(cell_candidate.z - (target_z_theoretical + actual_mesh_height))
            dz3 = abs(cell_candidate.z - (target_z_theoretical - actual_mesh_height))

            effective_dz = min(dz1, dz2, dz3)
            z_match_tolerance = self.hex_size * 0.75

            # Line 452
            if (effective_dz < min_dz_abs_effective and
                    effective_dz < z_match_tolerance):
                min_dz_abs_effective = effective_dz
                best_match = cell_candidate

        return best_match

    def get_all_cells(self) -> List[Cell]:
        """Retorna lista de todas las celdas en la malla."""
        return list(self.cells.values())


def compute_voronoi_neighbors(self, periodic_theta: bool = True) -> None:
        """
        Calcula los vecinos de Voronoi para cada celda
        considerando periodicidad circunferencial.

        Args:
            periodic_theta (bool):
            Considerar periodicidad en dirección theta.
        """
        try:
            from scipy.spatial import Voronoi
        except ImportError:
            logger.error(
                "Scipy no instalado. Voronoi no disponible."
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

        # Replicar puntos para periodicidad theta
        extended_points = points.copy()
        index_mapping = list(range(n_original))

        if periodic_theta:
            # Réplicas izquierda y derecha
            for i, (theta, z) in enumerate(points):
                # Line 478
                extended_points.append((theta - 2 * math.pi, z))
                index_mapping.append(i)
                # Line 478 (similar, for right replica)
                extended_points.append((theta + 2 * math.pi, z))
                index_mapping.append(i)

        # Convertir a array numpy
        points_array = np.array(extended_points)

        # Calcular diagrama de Voronoi
        vor = Voronoi(points_array)

        # Construir diccionario de vecinos
        neighbor_dict: Dict[int, Set[int]] = {i: set() for i in range(len(extended_points))}
        for ridge in vor.ridge_points:
            i, j = ridge
            neighbor_dict[i].add(j)
            neighbor_dict[j].add(i)
        
        # Procesar vecinos para cada celda original
        for orig_idx in range(n_original):
            cell = original_cells[orig_idx]
            voronoi_neighbors = set()
            
            # Considerar todas las réplicas del punto actual
            replica_indices = [orig_idx]
            if periodic_theta:
                # Line 533
                replica_indices.append(
                    orig_idx + n_original
                )  # Réplica izquierda
                # Line 534
                replica_indices.append(
                    orig_idx + 2 * n_original
                )  # Réplica derecha

            # Recopilar vecinos de todas las réplicas
            for rep_idx in replica_indices:
                if rep_idx < len(neighbor_dict):
                    for neighbor_ext_idx in neighbor_dict[rep_idx]:
                        neighbor_orig_idx = index_mapping[neighbor_ext_idx]
                        # Excluir auto-vecindad y vecinos duplicados
                        if neighbor_orig_idx != orig_idx:
                            voronoi_neighbors.add(neighbor_orig_idx)
            
            # Asignar vecinos como objetos Cell
            cell.voronoi_neighbors = [
                original_cells[i] for i in voronoi_neighbors
            ]

        logger.info(
            f"Vecinos Voronoi calculados para {n_original} celdas."
        )

def to_dict(self) -> Dict[str, Any]:
        """
        Retorna representación de la malla como diccionario.
        """
        return {
            "metadata": {
                "radius": self.radius,
                "height_segments": self.height_segments,
                # Line 651 in original problem, now ~587
                "circumference_segments_actual":
                    self.circumference_segments_actual,
                "hex_size": self.hex_size,
                "periodic_z": self.periodic_z,
                "num_cells": len(self.cells),
                "z_bounds": {"min": self.min_z, "max": self.max_z},
                "total_height_approx": self.total_height_approx,
                "previous_flux": self.previous_flux
            },
            "cells": [cell.to_dict() for cell in self.cells.values()]
        }
