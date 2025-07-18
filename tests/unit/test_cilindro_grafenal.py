# mi-proyecto/tests/unit/test_cilindro_grafenal.py
"""Pruebas unitarias para el módulo `cilindro_grafenal`.

Este módulo contiene pruebas para las clases `Cell` y `HexCylindricalMesh`,
así como funciones auxiliares relacionadas con mallas hexagonales cilíndricas
utilizadas en el contexto de simulaciones o modelado
(posiblemente con grafeno).

Las pruebas verifican la correcta inicialización de objetos, la funcionalidad
de los métodos principales (como obtención de celdas, vecinos, conversiones de
coordenadas) y la integridad estructural de las mallas generadas.
"""
import pytest
import numpy as np
import math
import logging
from watchers.watchers_tools.malla_watcher.utils.cilindro_grafenal import (
    Cell,
    HexCylindricalMesh,
    axial_to_cartesian_flat,
    cartesian_flat_to_cylindrical
)

logger = logging.getLogger(__name__)
# Opcional: configurar un handler básico si quieres ver estos logs durante
# la ejecución de pytest -s
# if not logger.handlers:
#     handler = logging.StreamHandler()
#     handler.setFormatter(
#         logging.Formatter('%(levelname)s:%(name)s:%(message)s')
#     )
#     logger.addHandler(handler)
#     logger.setLevel(logging.DEBUG) # o INFO

# --- Fixtures para Cell y HexCylindricalMesh ---
# (puedes mover/adaptar de test_malla_watcher.py)


@pytest.fixture
def sample_cell_cg():  # cg para cilindro_grafenal
    """Proporciona una instancia de `Cell` para pruebas.

    Esta celda de ejemplo está configurada con valores específicos
    para radio cilíndrico, theta, z, coordenadas axiales (q, r),
    amplitud, velocidad y vector q.

    Returns:
        Cell: Una instancia de `Cell` preconfigurada.
    """
    return Cell(
        cyl_radius=5.0, cyl_theta=np.pi/2, cyl_z=1.0, q_axial=1, r_axial=-1,
        amplitude=10.0, velocity=0.5, q_vector=np.array([0.2, -0.1])
    )


@pytest.fixture
def sample_mesh_cg():
    """Proporciona una instancia de `HexCylindricalMesh` para pruebas.

    Esta malla de ejemplo se inicializa con un radio, número de segmentos
    en altura, objetivo de segmentos en la circunferencia, tamaño de hexágono
    y periodicidad en Z definidos.

    Returns:
        HexCylindricalMesh:
        Una instancia de `HexCylindricalMesh` preconfigurada.
    """
    return HexCylindricalMesh(
        radius=3.0,
        height_segments=2,
        circumference_segments_target=12,
        hex_size=1.0,
        periodic_z=False
    )


# --- Tests para la Clase Cell ---
def test_cell_initialization(sample_cell_cg: Cell):
    """Verifica la inicialización correcta de los atributos de una `Cell`.

    Args:
        sample_cell_cg (Cell): Fixture que proporciona una instancia de `Cell`.
    """
    assert sample_cell_cg.r == 5.0
    assert sample_cell_cg.theta == np.pi/2
    assert sample_cell_cg.z == 1.0
    assert sample_cell_cg.q_axial == 1
    assert sample_cell_cg.r_axial == -1
    assert sample_cell_cg.amplitude == 10.0
    assert sample_cell_cg.velocity == 0.5
    # CAMBIADO: Verificar q_vector es un array NumPy
    assert isinstance(sample_cell_cg.q_vector, np.ndarray)
    assert sample_cell_cg.q_vector.shape == (2,)
    # Verificar valor
    assert np.array_equal(sample_cell_cg.q_vector, np.array([0.2, -0.1]))


def test_cell_repr(sample_cell_cg: Cell):
    """Prueba la representación en cadena de una instancia de `Cell`.

    Verifica que la salida de `repr(cell)` contenga la información esperada
    sobre las coordenadas axiales, cilíndricas, amplitud, velocidad y q_vector.

    Args:
        sample_cell_cg (Cell): Fixture que proporciona una instancia de `Cell`.
    """
    repr_str = repr(sample_cell_cg)
    assert "Cell(ax=(1,-1)" in repr_str
    assert "cyl=(r=5.00, θ=1.57, z=1.00)" in repr_str
    assert "amp=10.00" in repr_str
    assert "vel=0.50" in repr_str
    # CAMBIADO: Verificar q_vector en repr
    # Aceptar formatos de numpy.array repr
    assert "q_v=[0.20, -0.10]" in repr_str


def test_cell_to_dict(sample_cell_cg: Cell):
    """Prueba la conversión de una instancia de `Cell` a un diccionario.

    Verifica que el método `to_dict()` retorna un diccionario con las claves
    y valores correctos, incluyendo coordenadas axiales, cilíndricas,
    amplitud, velocidad y q_vector (como lista).

    Args:
        sample_cell_cg (Cell): Fixture que proporciona una instancia de `Cell`.
    """
    cell_dict = sample_cell_cg.to_dict()
    assert cell_dict["axial_coords"] == {"q": 1, "r": -1}
    assert cell_dict["cylindrical_coords"]["r"] == 5.0
    assert cell_dict["cylindrical_coords"]["theta"] == np.pi/2
    assert cell_dict["cylindrical_coords"]["z"] == 1.0
    assert cell_dict["amplitude"] == 10.0
    assert cell_dict["velocity"] == 0.5
    # CAMBIADO: Verificar q_vector como lista en dict
    assert isinstance(cell_dict["q_vector"], list)
    assert len(cell_dict["q_vector"]) == 2
    # Verificar valor como lista
    assert cell_dict["q_vector"] == [0.2, -0.1]


# --- Tests para la Clase HexCylindricalMesh ---


def test_mesh_initialization(sample_mesh_cg: HexCylindricalMesh):
    """Verifica la inicialización básica de una `HexCylindricalMesh`.

    Comprueba que los atributos principales como radio, segmentos de altura,
    tamaño de hexágono y periodicidad se establecen correctamente.
    También asegura que se generen celdas durante la inicialización y que
    la celda central (si existe) tenga las propiedades esperadas.

    Args:
        sample_mesh_cg (HexCylindricalMesh): Fixture que proporciona una
            instancia de `HexCylindricalMesh`.
    """
    assert sample_mesh_cg.radius == 3.0
    assert sample_mesh_cg.height_segments == 2
    assert sample_mesh_cg.hex_size == 1.0
    # Use 'is' for boolean comparison
    assert sample_mesh_cg.periodic_z is False

    assert len(sample_mesh_cg.cells) > 0, (
        "La inicialización debería crear celdas"
    )

    central_cell = sample_mesh_cg.get_cell(0, 0)
    if central_cell:
        assert isinstance(central_cell, Cell)
        assert central_cell.amplitude == 0.0
        assert central_cell.velocity == 0.0
        # CAMBIADO: Verificar q_vector inicial
        assert isinstance(central_cell.q_vector, np.ndarray)
        assert central_cell.q_vector.shape == (2,)
        assert np.array_equal(central_cell.q_vector, np.zeros(2))
    else:
        # This might happen if the mesh generation parameters in the fixture
        # don't result in a cell at (0,0). Log a warning but don't fail
        # the test setup.
        logger.warning(
            "Celda central (0,0) no encontrada en la malla de prueba."
        )
        # If the mesh is empty, the test will fail on the
        # len(sample_mesh.cells) > 0 assertion.


def test_mesh_get_cell(sample_mesh_cg: HexCylindricalMesh):
    """Prueba la obtención de celdas de la malla por coordenadas axiales.

    Verifica que `get_cell(q, r)` retorna una instancia de `Cell` para
    coordenadas existentes y `None` para coordenadas no existentes.
    El test puede omitirse si la celda (0,0) no existe en la malla de prueba.

    Args:
        sample_mesh_cg (HexCylindricalMesh): Fixture que proporciona una
            instancia de `HexCylindricalMesh`.
    """
    # Este test depende de que sample_mesh_cg genere celdas.
    # Si sample_mesh_cg falla en su setup (como en el log), este test no se
    # ejecuta.
    # Si sample_mesh_cg genera celdas pero no incluye (0,0), el skip es
    # correcto.
    q, r = 0, 0
    cell_00 = sample_mesh_cg.get_cell(q, r)
    if not cell_00:
        pytest.skip(
            f"Saltando test get_cell porque ({q},{r}) no existe."
        )

    assert isinstance(cell_00, Cell)

    cell_non_existent = sample_mesh_cg.get_cell(999, 999)
    assert cell_non_existent is None


def test_get_axial_neighbors_coords(
    sample_mesh_cg: HexCylindricalMesh
):  # Usar malla_watcher.HexCylindricalMesh
    """Verifica el cálculo de las coordenadas axiales de los vecinos.

    Prueba el método `get_axial_neighbors_coords(q, r)` para asegurar que
    retorna las 6 coordenadas axiales esperadas para una celda dada,
    independientemente de si las celdas vecinas existen realmente en la malla.

    Args:
        sample_mesh_cg (HexCylindricalMesh): Fixture que proporciona una
        instancia de `HexCylindricalMesh` (usada aquí principalmente
        para invocar el método, la lógica es independiente de la instancia).
    """
    # Este test no depende de las celdas reales, solo de la lógica de
    # get_axial_neighbors
    q, r = 2, 3
    # Coincide con tu original
    expected_neighbors = [
        (3, 3), (3, 2), (2, 2),  # q+1, r; q+1, r-1; q, r-1
        (1, 3), (1, 4), (2, 4)  # q-1, r; q-1, r+1; q, r+1
    ]
    actual_neighbors = sample_mesh_cg.get_axial_neighbors_coords(q, r)
    assert isinstance(actual_neighbors, list)
    assert len(actual_neighbors) == 6
    # Comparar
    assert set(actual_neighbors) == set(expected_neighbors)


def test_mesh_get_neighbor_cells(sample_mesh_cg: HexCylindricalMesh):
    """Prueba la obtención de las celdas vecinas existentes.

    Verifica que `get_neighbor_cells(q, r)` retorna una lista de instancias
    de `Cell` que son vecinas válidas de la celda en (q,r).
    El número de vecinos encontrados debe estar entre 0 y 6.
    El test puede omitirse si la celda (0,0) no existe en la malla de prueba.

    Args:
        sample_mesh_cg (HexCylindricalMesh): Fixture que proporciona una
            instancia de `HexCylindricalMesh`.
    """
    # Este test depende de que sample_mesh genere celdas y que
    # get_neighbor_cells funcione.
    # Si sample_mesh falla en su setup (como en el log), este test no se
    # ejecuta.
    q, r = 0, 0
    cell_00 = sample_mesh_cg.get_cell(q, r)
    if not cell_00:
        pytest.skip(
            f"Saltando test get_neighbor_cells porque ({q},{r}) no existe."
        )

    theoretical_neighbor_coords = set(
        sample_mesh_cg.get_axial_neighbors_coords(q, r)
    )
    neighbor_cells = sample_mesh_cg.get_neighbor_cells(q, r)
    assert isinstance(neighbor_cells, list)
    for cell in neighbor_cells:
        assert isinstance(cell, Cell)
        assert (cell.q_axial, cell.r_axial) in theoretical_neighbor_coords
    # El número exacto depende de la malla generada
    assert 0 <= len(neighbor_cells) <= 6


def test_mesh_verify_connectivity(sample_mesh_cg: HexCylindricalMesh):
    """Analiza la conectividad de la malla `HexCylindricalMesh` generada.

    Utiliza `verify_connectivity()` para obtener un reporte del número
    de vecinos por celda y verifica que este reporte cumpla con ciertas
    expectativas (e.g., no celdas aisladas, no más de 6 vecinos,
    proporciones razonables de celdas con diferente número de vecinos).

    Args:
        sample_mesh_cg (HexCylindricalMesh): Fixture que proporciona una
            instancia de `HexCylindricalMesh`.
    """
    # Este test depende de que sample_mesh_cg genere celdas y que
    # verify_connectivity funcione.
    # Si sample_mesh_cg falla en su setup (como en el log), este test no se
    # ejecuta.
    mesh = sample_mesh_cg
    assert len(mesh.cells) > 0, "La inicialización no generó celdas"
    connectivity_report = mesh.verify_connectivity()
    logger.info(
        f"Reporte de Conectividad para prueba: {connectivity_report}"
    )

    if connectivity_report:
        assert max(connectivity_report.keys()) <= 6, (
            "Se encontraron celdas con más de 6 vecinos"
        )

    assert connectivity_report.get(0, 0) == 0, (
        "Se encontraron celdas aisladas (0 vecinos)"
    )
    # Ajustar umbrales si es necesario según la fixture
    assert connectivity_report.get(1, 0) <= len(mesh.cells) * 0.15, (
        "Demasiadas celdas con solo 1 vecino"
    )
    assert connectivity_report.get(2, 0) <= len(mesh.cells) * 0.25, (
        "Demasiadas celdas con solo 2 vecinos"
    )

    if len(mesh.cells) > 10:
        assert 6 in connectivity_report or 5 in connectivity_report, (
            "Ninguna celda tiene 5 o 6 vecinos, revisar generación/tamaño"
        )
        count_5_6 = (connectivity_report.get(5, 0) +
                     connectivity_report.get(6, 0))
        assert count_5_6 >= len(mesh.cells) * 0.3, (
            "Menos del 30% de las celdas tienen 5 o 6 vecinos"
        )


def test_compute_voronoi_neighbors(sample_mesh_cg: HexCylindricalMesh):
    """Verifica que compute_voronoi_neighbors asigne vecinos ≤ 8."""
    if len(sample_mesh_cg.cells) < 3:
        pytest.skip("Se necesitan ≥3 celdas para Voronoi")
    sample_mesh_cg.compute_voronoi_neighbors(periodic_theta=True)
    for cell in sample_mesh_cg.get_all_cells():
        assert 0 <= len(cell.voronoi_neighbors) <= 8


def test_periodic_z_neighbors():
    """Comprueba vecinos envueltos en Z cuando periodic_z=True."""
    mesh = HexCylindricalMesh(
        radius=3.0,
        height_segments=2,
        circumference_segments_target=6,
        hex_size=1.0,
        periodic_z=True
    )
    if len(mesh.cells) < 3:
        pytest.skip("No se generaron suficientes celdas")
    central = mesh.get_cell(0, 0)
    if not central:
        pytest.skip("Celda (0,0) no encontrada")
    neighbors = mesh.get_neighbor_cells(0, 0)
    # Al menos debe encontrar vecinos envueltos en Z
    assert len(neighbors) >= 3


# --- Tests para Funciones Auxiliares ---


def test_axial_to_cartesian_flat():
    """Prueba la conversión de coordenadas axiales a cartesianas planas.

    Verifica que la función `axial_to_cartesian_flat(q, r, hex_size)`
    retorna las coordenadas (x, y) correctas para diferentes entradas
    de coordenadas axiales (q, r) y tamaño de hexágono.
    """
    assert axial_to_cartesian_flat(0, 0, 1.0) == (0.0, 0.0)
    # Añadir más casos de prueba con diferentes q, r y hex_size
    # Por ejemplo, q=1, r=0; q=0, r=1; q=1, r=1, etc.
    # q=1, r=0 -> x = 1.5, y = sqrt(3)/2
    assert axial_to_cartesian_flat(1, 0, 1.0) == pytest.approx(
        (1.5, math.sqrt(3) / 2)
    )


def test_cartesian_flat_to_cylindrical():
    """Prueba la conversión de coordenadas cartesianas planas a cilíndricas.

    Verifica que `cartesian_flat_to_cylindrical(x_flat, z_coord, radius)`
    retorna las coordenadas cilíndricas (r, theta, z) correctas.
    `x_flat` representa la coordenada 'x' desenrollada en el plano,
    `z_coord` es la altura 'z', y `radius` es el radio del cilindro.
    """
    radius = 5.0
    assert cartesian_flat_to_cylindrical(
        0.0, 0.0, radius
    ) == (radius, 0.0, 0.0)
    assert cartesian_flat_to_cylindrical(
        radius * math.pi, 10.0, radius
    ) == pytest.approx((radius, math.pi, 10.0))
    assert cartesian_flat_to_cylindrical(
        radius * 3 * math.pi, 5.0, radius
    ) == pytest.approx((radius, math.pi, 5.0))
    assert cartesian_flat_to_cylindrical(
        radius * -math.pi/2, -2.0, radius
    ) == pytest.approx((radius, 3 * math.pi / 2, -2.0))
