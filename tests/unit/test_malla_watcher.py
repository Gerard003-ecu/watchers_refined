# --- START OF FILE test_malla_watcher.py
# (REFINADO para Osciladores Acoplados) ---

import pytest
import os
import numpy as np
import math
from unittest.mock import patch, MagicMock

# Importaciones del módulo bajo prueba
from watchers.watchers_tools.malla_watcher.utils.cilindro_grafenal import (
    HexCylindricalMesh,
    Cell,
)
from watchers.watchers_tools.malla_watcher.malla_watcher import (
    PhosWave,
    Electron,
    simular_paso_malla,
    update_aggregate_state,
    map_cylinder_to_torus_coords,
    send_influence_to_torus,
    calculate_flux,
    app,
    simulation_loop,
    fetch_and_apply_torus_field,
)

# Configurar logging para pruebas
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """Fixture: Cliente de prueba para la aplicación Flask."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_requests_get():
    """Fixture: Mock para requests.get."""
    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher.requests.get"
    ) as mock_get:
        yield mock_get


@pytest.fixture
def mock_requests_post():
    """Fixture: Mock para requests.post."""
    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher.requests.post"
    ) as mock_post:
        yield mock_post


@pytest.fixture
def mock_mesh():
    """Fixture: Crea un mock de HexCylindricalMesh con celdas."""
    mesh = MagicMock(spec=HexCylindricalMesh)
    cell1 = Cell(cyl_radius=5, cyl_theta=0, cyl_z=0, q_axial=0, r_axial=0)
    cell2 = Cell(cyl_radius=5, cyl_theta=1, cyl_z=1, q_axial=1, r_axial=0)
    mesh.cells = {(0, 0): cell1, (1, 0): cell2}
    mesh.get_all_cells.return_value = [cell1, cell2]
    mesh.previous_flux = 0.0
    return mesh


@pytest.fixture
def malla_para_test_aplicar_campo():
    """Fixture: Malla para probar aplicación de campo externo."""
    return HexCylindricalMesh(
        radius=3.0,
        height_segments=1,
        circumference_segments_target=6,
        hex_size=1.0,
    )


@pytest.fixture
def reset_globals():
    """Fixture: Mocks global state for API tests."""
    # Patch de variables globales
    patcher_mesh = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".malla_cilindrica_global")
    patcher_resonador = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".resonador_global")
    patcher_electron = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".electron_global")
    patcher_agg_state = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".aggregate_state", new_callable=dict)
    patcher_agg_lock = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".aggregate_state_lock")
    patcher_ctrl_params = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".control_params", new_callable=dict)
    patcher_ctrl_lock = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".control_lock")

    # Patch de constantes
    patchers_const = {
        "MATRIZ_ECU_BASE_URL": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MATRIZ_ECU_BASE_URL", "http://mock-ecu:8000"),
        "TORUS_NUM_CAPAS": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".TORUS_NUM_CAPAS", 3),
        "TORUS_NUM_FILAS": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".TORUS_NUM_FILAS", 4),
        "TORUS_NUM_COLUMNAS": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".TORUS_NUM_COLUMNAS", 5),
        "AMPLITUDE_INFLUENCE_THRESHOLD": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".AMPLITUDE_INFLUENCE_THRESHOLD", 5.0),
        "MAX_AMPLITUDE_FOR_NORMALIZATION": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MAX_AMPLITUDE_FOR_NORMALIZATION", 20.0),
        "SIMULATION_INTERVAL": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".SIMULATION_INTERVAL", 0.5),
        "DPHI_DT_INFLUENCE_THRESHOLD": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".DPHI_DT_INFLUENCE_THRESHOLD", 1.0),
        "BASE_COUPLING_T": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".BASE_COUPLING_T", 0.6),
        "BASE_DAMPING_E": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".BASE_DAMPING_E", 0.1),
        "K_GAIN_COUPLING": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".K_GAIN_COUPLING", 0.1),
        "K_GAIN_DAMPING": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".K_GAIN_DAMPING", 0.05),
        "REQUESTS_TIMEOUT": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".REQUESTS_TIMEOUT", 2.0),
        "MESH_RADIUS": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MESH_RADIUS", 7.7),
        "MESH_HEIGHT_SEGMENTS": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MESH_HEIGHT_SEGMENTS", 2),
        "MESH_CIRCUMFERENCE_SEGMENTS": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MESH_CIRCUMFERENCE_SEGMENTS", 11),
        "MESH_HEX_SIZE": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MESH_HEX_SIZE", 0.8),
        "MESH_PERIODIC_Z": patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MESH_PERIODIC_Z", False),
    }
    started_patchers_const = {
        name: p.start() for name, p in patchers_const.items()
    }

    mock_mesh = patcher_mesh.start()
    mock_resonador = patcher_resonador.start()
    mock_electron = patcher_electron.start()
    mock_agg_state = patcher_agg_state.start()
    patcher_agg_lock.start()
    mock_ctrl_params = patcher_ctrl_params.start()
    patcher_ctrl_lock.start()

    # Configurar mock_mesh
    mock_mesh.cells = {
        (0, 0): Cell(cyl_radius=5.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0,
                     r_axial=0),
        (1, 0): Cell(cyl_radius=5.0, cyl_theta=0.5, cyl_z=0.0, q_axial=1,
                     r_axial=0),
        (0, 1): Cell(cyl_radius=5.0, cyl_theta=0.0, cyl_z=1.0, q_axial=0,
                     r_axial=1),
    }
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())
    mock_mesh.get_cell.side_effect = lambda q, r: mock_mesh.cells.get((q, r))
    mock_mesh.radius = 5.0
    mock_mesh.height_segments = 3
    mock_mesh.circumference_segments_actual = 6
    mock_mesh.hex_size = 1.0
    mock_mesh.periodic_z = False
    mock_mesh.min_z = 0.0
    mock_mesh.max_z = 1.0
    mock_mesh.verify_connectivity.return_value = {6: len(mock_mesh.cells)}
    mock_mesh.previous_flux = 0.0

    # Configurar mock_resonador y mock_electron
    mock_resonador.C = started_patchers_const["BASE_COUPLING_T"]
    mock_electron.D = started_patchers_const["BASE_DAMPING_E"]
    mock_resonador.ajustar_coeficientes.return_value = None
    mock_electron.ajustar_coeficientes.return_value = None

    # Configurar mock_agg_state
    mock_agg_state.update(
        {
            "avg_amplitude": 0.0,
            "max_amplitude": 0.0,
            "avg_velocity": 0.0,
            "max_velocity": 0.0,
            "avg_kinetic_energy": 0.0,
            "max_kinetic_energy": 0.0,
            "avg_activity_magnitude": 0.0,
            "max_activity_magnitude": 0.0,
            "cells_over_threshold": 0,
        }
    )

    # Configurar mock_ctrl_params
    mock_ctrl_params.update(
        {
            "phoswave_C": started_patchers_const["BASE_COUPLING_T"],
            "electron_D": started_patchers_const["BASE_DAMPING_E"],
        }
    )

    yield (
        mock_mesh,
        mock_resonador,
        mock_electron,
        mock_agg_state,
        mock_ctrl_params,
    )

    patcher_mesh.stop()
    patcher_resonador.stop()
    patcher_electron.stop()
    patcher_agg_state.stop()
    patcher_agg_lock.stop()
    patcher_ctrl_params.stop()
    patcher_ctrl_lock.stop()
    for p in patchers_const.values():
        p.stop()


# Tests para la Clase PhosWave
def test_phoswave_initialization():
    """Test: Inicialización correcta de atributos de PhosWave."""
    wave = PhosWave(coef_acoplamiento=0.7)
    assert wave.C == 0.7
    wave_neg = PhosWave(coef_acoplamiento=-0.5)
    assert wave_neg.C == 0.0


def test_phoswave_ajustar_coeficientes():
    """Test: Ajuste correcto del coeficiente de acoplamiento."""
    wave = PhosWave(coef_acoplamiento=0.7)
    wave.ajustar_coeficientes(0.9)
    assert wave.C == 0.9
    wave.ajustar_coeficientes(-0.2)
    assert wave.C == 0.0


# Tests para la Clase Electron
def test_electron_initialization():
    """Test: Inicialización correcta de atributos de Electron."""
    elec = Electron(coef_amortiguacion=0.3)
    assert elec.D == 0.3
    elec_neg = Electron(coef_amortiguacion=-0.1)
    assert elec_neg.D == 0.0


def test_electron_ajustar_coeficientes():
    """Test: Ajuste correcto del coeficiente de amortiguación."""
    elec = Electron(coef_amortiguacion=0.3)
    elec.ajustar_coeficientes(0.5)
    assert elec.D == 0.5
    elec.ajustar_coeficientes(-0.05)
    assert elec.D == 0.0


def test_apply_external_field_to_mesh_logic(malla_para_test_aplicar_campo):
    """Test: Aplicación de campo vectorial externo usando interpolación."""
    from watchers.watchers_tools.malla_watcher.malla_watcher import (
        apply_external_field_to_mesh as apply_ext_field_func,
    )

    mesh_instance = malla_para_test_aplicar_campo
    if not mesh_instance.cells:
        pytest.skip("Malla vacía, no se puede aplicar campo externo.")

    num_capas_torus, num_rows_torus, num_cols_torus = 2, 5, 7
    external_field_list = []
    for k in range(num_capas_torus):
        capa_data = []
        for i in range(num_rows_torus):
            row_data = []
            for j in range(num_cols_torus):
                row_data.append([float(j), float(i + k)])
            capa_data.append(row_data)
        external_field_list.append(capa_data)

    apply_ext_field_func(mesh_instance, external_field_list)

    changed_count = 0
    for coords, cell in mesh_instance.cells.items():
        assert isinstance(cell.q_vector, np.ndarray)
        assert cell.q_vector.shape == (2,)
        is_zero_vector = np.array_equal(cell.q_vector, np.zeros(2))
        if not is_zero_vector:
            changed_count += 1

    if len(mesh_instance.cells) > 0:
        assert changed_count > 0 or any(
            np.linalg.norm(c.q_vector) > 1e-9
            for c in mesh_instance.cells.values()
        ), (
            "apply_external_field_to_mesh no modificó ningún q_vector de "
            "forma no nula."
        )
    logger.info(
        f"{changed_count} de {len(mesh_instance.cells)} celdas actualizaron "
        "su q_vector a no-cero."
    )


# Tests para la Lógica de Simulación
@pytest.fixture
def mock_malla_sim():
    """
    Fixture: Configura una pequeña malla mockeada para simulación.
    """
    cell_center = Cell(
        5.0, 0.0, 0.0, 0, 0, amplitude=10.0, velocity=0.0,
        q_vector=np.array([0.1, 0.2])
    )
    cell_neighbor1 = Cell(
        5.0, 0.5, 0.0, 1, 0, amplitude=0.0, velocity=0.0,
        q_vector=np.array([-0.1, -0.2])
    )
    cell_neighbor2 = Cell(
        5.0, -0.5, 0.0, -1, 0, amplitude=0.0, velocity=0.0,
        q_vector=np.array([0.0, 0.0])
    )

    mock_mesh = MagicMock(spec=HexCylindricalMesh)
    mock_mesh.cells = {
        (0, 0): cell_center,
        (1, 0): cell_neighbor1,
        (-1, 0): cell_neighbor2
    }
    mock_mesh.get_all_cells.return_value = [
        cell_center, cell_neighbor1, cell_neighbor2
    ]

    def mock_get_neighbors(q, r):
        if (q, r) == (0, 0):
            return [cell_neighbor1, cell_neighbor2]
        elif (q, r) == (1, 0):
            return [cell_center]
        elif (q, r) == (-1, 0):
            return [cell_center]
        return []

    mock_mesh.get_neighbor_cells.side_effect = mock_get_neighbors
    mock_mesh.get_cell.side_effect = lambda q, r: mock_mesh.cells.get((q, r))
    mock_mesh.previous_flux = 0.0

    mock_resonador = MagicMock(spec=PhosWave)
    mock_resonador.C = 0.5

    mock_electron = MagicMock(spec=Electron)
    mock_electron.D = 0.1

    mock_sim_interval = 0.1

    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".malla_cilindrica_global", mock_mesh), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".resonador_global", mock_resonador), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".electron_global", mock_electron), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".SIMULATION_INTERVAL", mock_sim_interval):
        yield (
            cell_center,
            cell_neighbor1,
            cell_neighbor2,
            mock_mesh,
            mock_resonador,
            mock_electron,
            mock_sim_interval,
        )


def test_simular_paso_malla_propagation(mock_malla_sim):
    """Test: simular_paso_malla propaga amplitud/velocidad correctamente."""
    (
        cell_center,
        cell_neighbor1,
        cell_neighbor2,
        mock_mesh,
        mock_resonador,
        mock_electron,
        dt,
    ) = mock_malla_sim

    cell_center.amplitude = 10.0
    cell_center.velocity = 0.0
    cell_center.q_vector = np.array([0.1, 0.2])
    cell_neighbor1.amplitude = 0.0
    cell_neighbor1.velocity = 0.0
    cell_neighbor1.q_vector = np.array([-0.1, -0.2])
    cell_neighbor2.amplitude = 0.0
    cell_neighbor2.velocity = 0.0
    cell_neighbor2.q_vector = np.array([0.0, 0.0])

    mock_resonador.C = 0.5
    mock_electron.D = 0.0

    norm_q_center = np.linalg.norm(cell_center.q_vector)
    modulated_C_center = mock_resonador.C * max(0.0, 1.0 + norm_q_center)
    expected_next_vel_c = -20 * modulated_C_center * dt
    expected_next_amp_c = (
        cell_center.amplitude + expected_next_vel_c * dt
    )

    norm_q_n1 = np.linalg.norm(cell_neighbor1.q_vector)
    modulated_C_n1 = mock_resonador.C * max(0.0, 1.0 + norm_q_n1)
    F_coupling_n1 = modulated_C_n1 * (
        cell_center.amplitude - cell_neighbor1.amplitude
    )
    expected_next_vel_n1 = F_coupling_n1 * dt
    expected_next_amp_n1 = (
        cell_neighbor1.amplitude + expected_next_vel_n1 * dt
    )

    norm_q_n2 = np.linalg.norm(cell_neighbor2.q_vector)
    modulated_C_n2 = mock_resonador.C * max(0.0, 1.0 + norm_q_n2)
    F_coupling_n2 = modulated_C_n2 * (
        cell_center.amplitude - cell_neighbor2.amplitude
    )
    expected_next_vel_n2 = F_coupling_n2 * dt
    expected_next_amp_n2 = (
        cell_neighbor2.amplitude + expected_next_vel_n2 * dt
    )

    simular_paso_malla()

    assert cell_center.amplitude == pytest.approx(expected_next_amp_c)
    assert cell_center.velocity == pytest.approx(expected_next_vel_c)
    assert cell_neighbor1.amplitude == pytest.approx(expected_next_amp_n1)
    assert cell_neighbor1.velocity == pytest.approx(expected_next_vel_n1)
    assert cell_neighbor2.amplitude == pytest.approx(expected_next_amp_n2)
    assert cell_neighbor2.velocity == pytest.approx(expected_next_vel_n2)


def test_simular_paso_malla_damping(mock_malla_sim):
    """Test: simular_paso_malla aplica amortiguación correctamente."""
    (
        cell_center,
        cell_neighbor1,
        cell_neighbor2,
        mock_mesh,
        mock_resonador,
        mock_electron,
        dt,
    ) = mock_malla_sim

    cell_center.amplitude = 10.0
    cell_center.velocity = 5.0
    cell_center.q_vector = np.zeros(2)
    cell_neighbor1.amplitude = 0.0
    cell_neighbor1.velocity = 0.0
    cell_neighbor1.q_vector = np.zeros(2)
    cell_neighbor2.amplitude = 0.0
    cell_neighbor2.velocity = 0.0
    cell_neighbor2.q_vector = np.zeros(2)

    mock_resonador.C = 0.0
    mock_electron.D = 0.2

    expected_next_vel_c = 5.0 + (-mock_electron.D * 5.0) * dt
    expected_next_amp_c = 10.0 + expected_next_vel_c * dt

    expected_next_amp_n1 = 0.0
    expected_next_vel_n1 = 0.0
    expected_next_amp_n2 = 0.0
    expected_next_vel_n2 = 0.0

    simular_paso_malla()

    assert cell_center.amplitude == pytest.approx(expected_next_amp_c)
    assert cell_center.velocity == pytest.approx(expected_next_vel_c)
    assert cell_neighbor1.amplitude == pytest.approx(expected_next_amp_n1)
    assert cell_neighbor1.velocity == pytest.approx(expected_next_vel_n1)
    assert cell_neighbor2.amplitude == pytest.approx(expected_next_amp_n2)
    assert cell_neighbor2.velocity == pytest.approx(expected_next_vel_n2)


def test_calculate_flux():
    """Test: calculate_flux suma la componente vy de q_vector."""
    mesh = HexCylindricalMesh(
        radius=1.0, height_segments=1,
        circumference_segments_target=3, hex_size=1.0
    )
    mesh.cells.clear()
    cell1 = Cell(
        cyl_radius=1.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0, r_axial=0,
        q_vector=np.array([1.0, 2.0]))
    cell2 = Cell(
        cyl_radius=1.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=1, r_axial=0,
        q_vector=np.array([-0.5, 3.0]))
    cell3 = Cell(
        cyl_radius=1.0, cyl_theta=0.0, cyl_z=1.0, q_axial=0, r_axial=1,
        q_vector=np.array([0.0, -1.5]))
    mesh.cells[(0, 0)] = cell1
    mesh.cells[(1, 0)] = cell2
    mesh.cells[(0, 1)] = cell3
    expected_flux = 2.0 + 3.0 + (-1.5)
    actual_flux = calculate_flux(mesh)
    assert actual_flux == pytest.approx(expected_flux)
    mesh_empty = HexCylindricalMesh(
        radius=1.0, height_segments=1,
        circumference_segments_target=3, hex_size=1.0
    )
    mesh_empty.cells.clear()
    assert calculate_flux(mesh_empty) == pytest.approx(0.0)


def test_dphi_dt_calculation(mock_malla_sim):
    """Test: Cálculo de dPhi/dt en simulation_loop."""
    (
        cell_center,
        cell_neighbor1,
        cell_neighbor2,
        mock_mesh,
        mock_resonador,
        mock_electron,
        dt,
    ) = mock_malla_sim

    # Define patchers individually
    with (
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.calculate_flux")
        as mock_calculate_flux,
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.fetch_and_apply_torus_field"),
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.simular_paso_malla"),
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.update_aggregate_state"),
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.send_influence_to_torus")
        as mock_send_influence,
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.stop_simulation_event")
        as mock_stop_event
    ):

        mock_mesh.previous_flux = 10.0
        mock_calculate_flux.return_value = 15.0
        mock_stop_event.is_set.side_effect = [False, True]
        mock_stop_event.wait.return_value = None

        with patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher.SIMULATION_INTERVAL", dt
        ):
            simulation_loop()

        expected_dphi_dt_step1 = (15.0 - 10.0) / dt
        assert mock_mesh.previous_flux == pytest.approx(15.0)
        mock_calculate_flux.assert_called_once_with(mock_mesh)
        mock_send_influence.assert_called_once_with(
            pytest.approx(expected_dphi_dt_step1)
        )


def test_send_influence_to_torus(mock_requests_post):
    """Test: send_influence_to_torus envía POST a ECU con dPhi/dt."""
    mock_post = mock_requests_post
    dphi_dt_value = 7.5
    with (
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_FILAS", 10),
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_COLUMNAS", 15),
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.MATRIZ_ECU_BASE_URL", "http://mock-ecu:8000") as mock_base_url,
        patch("watchers.watchers_tools.malla_watcher.malla_watcher.REQUESTS_TIMEOUT", 2.0)
    ):
        expected_target_capa = 0
        expected_target_row = 10 // 2
        expected_target_col = 15 // 2
        send_influence_to_torus(dphi_dt_value)
        expected_url = f"{mock_base_url.new}/api/ecu/influence"
        watcher_name = f"malla_watcher_dPhiDt{dphi_dt_value:.3f}"
        expected_json = {
            "capa": expected_target_capa,
            "row": expected_target_row,
            "col": expected_target_col,
            "vector": [dphi_dt_value, 0.0],
            "nombre_watcher": watcher_name,
        }
        mock_post.assert_called_once_with(
            expected_url,
            json=expected_json,
            timeout=pytest.approx(2.0)
        )


def test_fetch_and_apply_torus_field(mock_requests_get):
    """Test: fetch_and_apply_torus_field obtiene campo vectorial y aplica."""
    mock_get = mock_requests_get
    mock_ecu_field_data = [
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
    ]
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_ecu_field_data
    mock_get.return_value.raise_for_status.return_value = None

    with (
        patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher.malla_cilindrica_global"
        ) as mock_mesh_global_instance,
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher.apply_external_field_to_mesh"
        ) as mock_apply_func,
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher.MATRIZ_ECU_BASE_URL", "http://mock-ecu:8000"
        ) as mock_base_url,
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher.REQUESTS_TIMEOUT", 3.0):
    )
        mock_mesh_global_instance.configure_mock(
            cells={(0, 0): "dummy_cell"})
        fetch_and_apply_torus_field()
        expected_url = f"{mock_base_url.new}/api/ecu/field_vector"
        mock_get.assert_called_once_with(
            expected_url,
            timeout=pytest.approx(3.0))
        mock_get.return_value.raise_for_status.assert_called_once()
        mock_apply_func.assert_called_once_with(
            mock_mesh_global_instance, mock_ecu_field_data)


# Tests para Estado Agregado
@pytest.fixture
def mock_malla_state():
    """Configura malla mockeada con amplitudes para estado agregado."""
    cell1 = Cell(
        5.0, 0.0, 0.0, 0, 0, amplitude=10.0, velocity=1.0,
        q_vector=np.array([0.1, 0.2]))
    cell2 = Cell(5.0, 0.5, 0.0, 1, 0, amplitude=-5.0, velocity=-2.0,
                 q_vector=np.array([-0.1, -0.2]))
    cell3 = Cell(5.0, 1.0, 0.0, 2, 0, amplitude=2.0, velocity=0.1,
                 q_vector=np.array([0.0, 0.0]))
    cell4 = Cell(5.0, 1.5, 0.0, 3, 0, amplitude=6.0, velocity=3.0,
                 q_vector=np.array([0.3, 0.4]))

    mock_mesh = MagicMock(spec=HexCylindricalMesh)
    mock_mesh.cells = {
        (0, 0): cell1,
        (1, 0): cell2,
        (2, 0): cell3,
        (3, 0): cell4
    }
    mock_mesh.get_all_cells.return_value = [cell1, cell2, cell3, cell4]
    mock_mesh.previous_flux = 0.0

    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher.malla_cilindrica_global", mock_mesh
        ),
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher.AMPLITUDE_INFLUENCE_THRESHOLD", 5.0
        ),
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher.DPHI_DT_INFLUENCE_THRESHOLD", 1.0
        ),
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher.SIMULATION_INTERVAL", 0.5
        ),
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher.aggregate_state_lock"
        ):
        yield mock_mesh, [
            cell1,
            cell2,
            cell3,
            cell4,
        ]


def test_update_aggregate_state(mock_malla_state):
    """Test: update_aggregate_state calcula métricas basadas en amplitud."""
    mock_mesh, cells = mock_malla_state

    expected_avg_amp = (10.0 - 5.0 + 2.0 + 6.0) / 4.0
    expected_max_amp = 10.0
    expected_avg_vel = (1.0 - 2.0 + 0.1 + 3.0) / 4.0
    expected_max_vel = 3.0
    # KE = 0.5 * m * v^2. Assuming m=1 for simplicity in test
    ke_values = [0.5 * 1.0**2, 0.5 * (-2.0)**2, 0.5 * 0.1**2, 0.5 * 3.0**2]
    expected_avg_ke = sum(ke_values) / 4.0
    expected_max_ke = max(ke_values)

    activity_mags = []
    for cell_obj in cells:
        # activity_magnitude = sqrt(amplitude^2 + velocity^2)
        act_mag = math.sqrt(cell_obj.amplitude**2 + cell_obj.velocity**2)
        activity_mags.append(act_mag)

    expected_avg_activity = sum(activity_mags) / 4.0
    expected_max_activity = max(activity_mags)
    # cells_over_threshold: amplitude > AMPLITUDE_INFLUENCE_THRESHOLD (5.0)
    # cells are 10.0, -5.0 (abs is 5.0, not >), 2.0, 6.0
    expected_over_thresh = 2  # Only 10.0 and 6.0

    update_aggregate_state()

    from watchers.watchers_tools.malla_watcher.malla_watcher import (
        aggregate_state)

    assert aggregate_state["avg_amplitude"] == pytest.approx(expected_avg_amp)
    assert aggregate_state["max_amplitude"] == pytest.approx(expected_max_amp)
    assert aggregate_state["avg_velocity"] == pytest.approx(expected_avg_vel)
    assert aggregate_state["max_velocity"] == pytest.approx(expected_max_vel)
    assert aggregate_state["avg_kinetic_energy"] == pytest.approx(expected_avg_ke)  # noqa: E501
    assert aggregate_state["max_kinetic_energy"] == pytest.approx(expected_max_ke)  # noqa: E501
    assert aggregate_state["avg_activity_magnitude"] == pytest.approx(expected_avg_activity)  # noqa: E501
    assert aggregate_state["max_activity_magnitude"] == pytest.approx(expected_max_activity)  # noqa: E501
    assert aggregate_state["cells_over_threshold"] == expected_over_thresh

    # Test case for empty mesh
    mock_mesh_empty = MagicMock(spec=HexCylindricalMesh)
    mock_mesh_empty.cells = {}
    mock_mesh_empty.get_all_cells.return_value = []

    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".malla_cilindrica_global", mock_mesh_empty), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".aggregate_state_lock"):
        update_aggregate_state()
        assert aggregate_state["avg_amplitude"] == 0.0
        assert aggregate_state["max_amplitude"] == 0.0
        assert aggregate_state["avg_velocity"] == 0.0
        assert aggregate_state["max_velocity"] == 0.0
        assert aggregate_state["avg_kinetic_energy"] == 0.0
        assert aggregate_state["max_kinetic_energy"] == 0.0
        assert aggregate_state["avg_activity_magnitude"] == 0.0
        assert (aggregate_state["max_activity_magnitude"] == 0.0)
        assert aggregate_state["cells_over_threshold"] == 0


# Tests para Mapeo a Toroide
@pytest.fixture
def mock_malla_map():
    """Fixture: Configura una malla mockeada para pruebas de mapeo."""
    mock_mesh = MagicMock(spec=HexCylindricalMesh)
    mock_mesh.cells = {}
    mock_mesh.min_z = -10.0
    mock_mesh.max_z = 10.0
    mock_mesh.circumference_segments_actual = 12
    mock_mesh.previous_flux = 0.0

    mock_torus_capas = 5
    mock_torus_filas = 10
    mock_torus_columnas = 15

    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".malla_cilindrica_global", mock_mesh), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".TORUS_NUM_CAPAS", mock_torus_capas), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".TORUS_NUM_FILAS", mock_torus_filas), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".TORUS_NUM_COLUMNAS", mock_torus_columnas), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MAX_AMPLITUDE_FOR_NORMALIZATION", 20.0):
        yield (mock_mesh, mock_torus_capas, mock_torus_filas,
               mock_torus_columnas)


def test_map_cylinder_to_torus_coords(mock_malla_map):
    """Test: map_cylinder_to_torus_coords mapea correctamente."""
    mock_mesh, num_capas, num_filas, num_columnas = mock_malla_map

    cell = Cell(
        cyl_radius=5.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=10, r_axial=20,
        amplitude=10.0, velocity=1.0, q_vector=np.array([0.0, 0.0]))
    mock_mesh.cells[(-99, -99)] = Cell(
        cyl_radius=0.0, cyl_theta=0.0, cyl_z=0.0, q_axial=-99, r_axial=-99)

    expected_coords = (2, 4, 7)  # Original: (2, 4, 7)
    actual_coords = map_cylinder_to_torus_coords(cell)
    assert actual_coords == expected_coords

    cell_zero_act = Cell(
        cyl_radius=5.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=10, r_axial=20,
        amplitude=0.0, velocity=0.0, q_vector=np.array([0.0, 0.0]))
    # activity_magnitude = 0. capa_norm = 0.
    # capa_idx = floor((1-0)*4 + 0.5) = floor(4.5) = 4
    expected_coords_zero = (4, 4, 7)
    actual_coords_zero = map_cylinder_to_torus_coords(cell_zero_act)
    assert actual_coords_zero == expected_coords_zero

    cell_max_act = Cell(
        cyl_radius=5.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=10, r_axial=20,
        amplitude=20.0, velocity=0.0, q_vector=np.array([0.0, 0.0]))
    # activity_magnitude = 20. capa_norm = 1.
    # capa_idx = floor((1-1)*4 + 0.5) = floor(0.5) = 0
    expected_coords_max = (0, 4, 7)
    actual_coords_max = map_cylinder_to_torus_coords(cell_max_act)
    assert actual_coords_max == expected_coords_max

    cell_high_act = Cell(
        cyl_radius=5.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=10, r_axial=20,
        amplitude=15.0, velocity=15.0, q_vector=np.array([0.0, 0.0]))
    # activity_magnitude = sqrt(15^2+15^2) =
    # sqrt(225+225)=sqrt(450) approx 21.2
    # capa_norm = min(21.2, 20.0) / 20.0 = 1.0
    # capa_idx = floor((1-1)*4 + 0.5) = 0
    expected_coords_high = (0, 4, 7)
    actual_coords_high = map_cylinder_to_torus_coords(cell_high_act)
    assert actual_coords_high == expected_coords_high


# Tests para Envío de Influencia
@pytest.fixture
def mock_send_influence():
    """Fixture: Configura mocks para probar send_influence_to_torus."""
    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher.requests.post"
    ) as mock_post, \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".MATRIZ_ECU_BASE_URL", "http://mock-ecu:8000"), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".TORUS_NUM_FILAS", 10), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".TORUS_NUM_COLUMNAS", 15), \
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher"
            ".REQUESTS_TIMEOUT", 2.0):
        mock_post.return_value.raise_for_status.return_value = None
        yield mock_post


# Tests de API


def test_api_health(client, reset_globals):
    """Test: Endpoint /api/health retorna estado y detalles de la malla."""
    mock_mesh, _, _, _, _ = reset_globals
    mock_mesh.cells = {
        (0, 0): Cell(1.0, 1.0, 1.0, 0, 0),
        (1, 0): Cell(1.0, 1.0, 1.0, 1, 0),
        (0, 1): Cell(1.0, 1.0, 1.0, 0, 1),
    }
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())
    mock_mesh.verify_connectivity.side_effect = (
        lambda: {6: len(mock_mesh.cells)})

    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher.simulation_thread"
    ) as mock_sim_thread:
        mock_sim_thread.is_alive.return_value = True
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["module"] == "Malla_watcher"
        assert data["message"] == "Malla_watcher operativo."
        details = data["details"]
        assert details["mesh"]["initialized"] is True
        assert details["mesh"]["num_cells"] == 3
        assert details["mesh"]["connectivity_status"] == "ok"
        assert details["mesh"]["min_neighbors"] == 6
        assert details["mesh"]["max_neighbors"] == 6
        assert details["mesh"]["z_periodic"] == mock_mesh.periodic_z
        assert details["resonator_simulation"]["running"] is True


def test_api_health_no_simulation_thread(client, reset_globals):
    """Test: /api/health cuando el hilo de simulación no está activo."""
    mock_mesh, _, _, _, _ = reset_globals
    mock_mesh.cells = {(0, 0): Cell(1.0, 1.0, 1.0, 0, 0)}
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())

    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "simulation_thread") as mock_sim_thread:
        mock_sim_thread.is_alive.return_value = False
        response = client.get("/api/health")
        assert response.status_code == 500
        data = response.get_json()
        assert data["status"] == "error"
        assert "Hilo de simulación del resonador inactivo" in data["message"]
        assert data["details"]["resonator_simulation"]["running"] is False


def test_api_health_empty_mesh(client, reset_globals):
    """Test: /api/health cuando la malla está inicializada pero vacía."""
    mock_mesh, _, _, _, _ = reset_globals
    mock_mesh.cells = {}
    mock_mesh.get_all_cells.return_value = []

    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "simulation_thread") as mock_sim_thread:
        mock_sim_thread.is_alive.return_value = True
        response = client.get("/api/health")
        assert response.status_code == 500
        data = response.get_json()
        assert data["status"] == "error"
        assert (
            "Malla inicializada pero contiene 0 celdas" in data["message"])
        assert (
            data["details"]["mesh"]["num_cells"] == 0)


def test_api_state(client, reset_globals):
    """Test: /api/state retorna estado agregado y parámetros de control."""
    (
        mock_mesh,
        mock_resonador,
        mock_electron,
        mock_agg_state,
        mock_ctrl_params,
    ) = reset_globals
    mock_agg_state.update(
        {
            "avg_amplitude": 12.3,
            "max_amplitude": 45.6,
            "avg_velocity": 1.5,
            "max_velocity": 3.0,
            "avg_kinetic_energy": 0.75,
            "max_kinetic_energy": 4.5,
            "avg_activity_magnitude": 13.0,
            "max_activity_magnitude": 46.0,
            "cells_over_threshold": 7,
        }
    )
    mock_mesh.cells = {
        (0, 0): Cell(5.0, 0.0, 0.0, 0, 0),
        (1, 0): Cell(5.0, 0.5, 0.0, 1, 0),
        (0, 1): Cell(5.0, 0.0, 1.0, 0, 1),
    }
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())
    expected_num_cells = len(mock_mesh.cells)
    expected_C = 0.75
    expected_D = 0.25
    mock_resonador.C = expected_C
    mock_electron.D = expected_D
    mock_ctrl_params.update(
        {"phoswave_C": expected_C, "electron_D": expected_D})

    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    state_data = data["state"]
    assert state_data["avg_amplitude"] == pytest.approx(12.3)
    assert state_data["max_amplitude"] == pytest.approx(45.6)
    assert state_data["avg_velocity"] == pytest.approx(1.5)
    assert state_data["max_velocity"] == pytest.approx(3.0)
    assert state_data["avg_kinetic_energy"] == pytest.approx(0.75)
    assert state_data["max_kinetic_energy"] == pytest.approx(4.5)
    assert state_data["avg_activity_magnitude"] == pytest.approx(13.0)
    assert state_data["max_activity_magnitude"] == pytest.approx(46.0)
    assert state_data["cells_over_threshold"] == 7
    assert state_data["num_cells"] == expected_num_cells
    assert (
        state_data["control_params"]["phoswave_C"] == pytest.approx(expected_C)
    )
    assert state_data["control_params"]["electron_D"] == pytest.approx(expected_D)  # noqa: E501


def test_api_control_success(client, reset_globals):
    """Test: /api/control ajusta parámetros de PhosWave y Electron."""
    _, mock_resonador, mock_electron, _, mock_ctrl_params = reset_globals
    from watchers.watchers_tools.malla_watcher.malla_watcher import (
        BASE_COUPLING_T,
        BASE_DAMPING_E,
        K_GAIN_COUPLING,
        K_GAIN_DAMPING,
    )

    mock_resonador.ajustar_coeficientes.side_effect = lambda val: setattr(
        mock_resonador, "C", max(0.0, val))
    mock_electron.ajustar_coeficientes.side_effect = lambda val: setattr(
        mock_electron, "D", max(0.0, val))

    signal_pos = 10.0
    expected_C_pos = BASE_COUPLING_T + K_GAIN_COUPLING * signal_pos
    expected_D_pos = BASE_DAMPING_E - K_GAIN_DAMPING * signal_pos

    response_pos = client.post(
        "/api/control", json={"control_signal": signal_pos})
    assert response_pos.status_code == 200
    data_pos = response_pos.get_json()
    assert data_pos["status"] == "success"
    assert "Parámetros ajustados" in data_pos["message"]
    mock_resonador.ajustar_coeficientes.assert_called_with(
        pytest.approx(max(0.0, expected_C_pos)))
    mock_electron.ajustar_coeficientes.assert_called_with(
        pytest.approx(max(0.0, expected_D_pos)))
    assert mock_resonador.C == pytest.approx(max(0.0, expected_C_pos))
    assert mock_electron.D == pytest.approx(max(0.0, expected_D_pos))
    assert mock_ctrl_params["phoswave_C"] == pytest.approx(max(0.0, expected_C_pos))  # noqa: E501
    assert mock_ctrl_params["electron_D"] == pytest.approx(max(0.0, expected_D_pos))  # noqa: E501

    mock_resonador.ajustar_coeficientes.reset_mock()
    mock_electron.ajustar_coeficientes.reset_mock()
    mock_resonador.C = BASE_COUPLING_T
    mock_electron.D = BASE_DAMPING_E
    mock_ctrl_params.update(
        {"phoswave_C": BASE_COUPLING_T, "electron_D": BASE_DAMPING_E})

    signal_neg = -5.0
    expected_C_neg = BASE_COUPLING_T + K_GAIN_COUPLING * signal_neg
    expected_D_neg = BASE_DAMPING_E - K_GAIN_DAMPING * signal_neg

    response_neg = client.post(
        "/api/control", json={"control_signal": signal_neg})
    assert response_neg.status_code == 200
    data_neg = response_neg.get_json()
    assert data_neg["status"] == "success"
    mock_resonador.ajustar_coeficientes.assert_called_with(
        pytest.approx(max(0.0, expected_C_neg)))
    mock_electron.ajustar_coeficientes.assert_called_with(
        pytest.approx(max(0.0, expected_D_neg)))
    assert mock_resonador.C == pytest.approx(max(0.0, expected_C_neg))
    assert mock_electron.D == pytest.approx(max(0.0, expected_D_neg))
    assert mock_ctrl_params["phoswave_C"] == pytest.approx(max(0.0, expected_C_neg))  # noqa: E501
    assert mock_ctrl_params["electron_D"] == pytest.approx(max(0.0, expected_D_neg))  # noqa: E501


def test_api_control_invalid_input(client):
    """Test: /api/control maneja JSON inválido o campos faltantes."""
    response_no_json = client.post(
        "/api/control",
        data="no es json",
        content_type="application/json")
    assert response_no_json.status_code == 400
    assert ("Payload JSON vacío, inválido o falta" in
            response_no_json.get_json()["message"])

    response_missing_field = client.post(
        "/api/control", json={"otro_campo": 123})
    assert response_missing_field.status_code == 400
    assert "falta 'control_signal'" in response_missing_field.get_json()["message"]  # noqa: E501

    response_wrong_type = client.post(
        "/api/control", json={"control_signal": "no_numero"})
    assert response_wrong_type.status_code == 400
    assert "El campo 'control_signal' debe ser un número" in response_wrong_type.get_json()["message"]  # noqa: E501


def test_api_event_pulse(client, reset_globals):
    """Test: /api/event aplica correctamente un pulso a la velocidad."""
    mock_mesh, _, _, _, _ = reset_globals
    cell_coords_q, cell_coords_r = 0, 0
    initial_velocity = 1.5
    pulse_magnitude = 5.0
    target_cell = Cell(
        cyl_radius=1.0,
        cyl_theta=1.0,
        cyl_z=1.0,
        q_axial=cell_coords_q,
        r_axial=cell_coords_r,
        velocity=initial_velocity)
    mock_mesh.cells = {(cell_coords_q, cell_coords_r): target_cell}
    mock_mesh.get_cell.side_effect = lambda q, r: mock_mesh.cells.get((q, r))

    payload = {
        "type": "pulse",
        "coords": {"q": cell_coords_q, "r": cell_coords_r},
        "magnitude": pulse_magnitude}
    response = client.post("/api/event", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["message"] == "Evento procesado"
    assert target_cell.velocity == pytest.approx(
        initial_velocity + pulse_magnitude)


def test_api_event_pulse_cell_not_found(client, reset_globals):
    """Test: /api/event maneja el caso de celda no encontrada."""
    mock_mesh, _, _, _, _ = reset_globals
    mock_mesh.cells = {}
    mock_mesh.get_cell.return_value = None
    payload = {
        "type": "pulse", "coords": {"q": 99, "r": 99}, "magnitude": 1.0}
    response = client.post("/api/event", json=payload)
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "warning"
    assert "Malla no inicializada o vacía" in data["message"]


def test_api_event_pulse_cell_not_found_in_populated_mesh(
        client, reset_globals):
    """Test: /api/event devuelve 404 si la celda no existe."""
    mock_mesh, _, _, _, _ = reset_globals
    existing_cell_coords_q, existing_cell_coords_r = 0, 0
    existing_cell = Cell(
        cyl_radius=1.0,
        cyl_theta=0.0,
        cyl_z=0.0,
        q_axial=existing_cell_coords_q,
        r_axial=existing_cell_coords_r)
    mock_mesh.cells = {(
        existing_cell_coords_q, existing_cell_coords_r): existing_cell}
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())

    coords_to_find_q, coords_to_find_r = 99, 99

    def get_cell_side_effect(q, r):
        if (q, r) == (existing_cell_coords_q, existing_cell_coords_r):
            return mock_mesh.cells.get((q, r))
        elif (q, r) == (coords_to_find_q, coords_to_find_r):
            return None
        return mock_mesh.cells.get((q, r))  # pragma: no cover

    mock_mesh.get_cell.side_effect = get_cell_side_effect

    payload = {
        "type": "pulse",
        "coords": {"q": coords_to_find_q, "r": coords_to_find_r},
        "magnitude": 1.0}
    response = client.post("/api/event", json=payload)
    assert response.status_code == 404
    data = response.get_json()
    assert data["status"] == "warning"
    message = f"Celda ({coords_to_find_q},{coords_to_find_r}) no encontrada"
    assert message in data["message"]
    assert existing_cell.velocity == 0.0


def test_api_malla(client, reset_globals):
    """Test: /api/malla devuelve la estructura completa de la malla."""
    mock_mesh, _, _, _, _ = reset_globals
    cell_a = Cell(mock_mesh.radius, 0.1, 0.5, 0, 0, amplitude=1,
                  velocity=0.1, q_vector=np.array([0.1, 0.1]))
    cell_b = Cell(mock_mesh.radius, 0.2, 1.5, 1, 0, amplitude=2,
                  velocity=0.2, q_vector=np.array([0.2, 0.2]))
    mock_mesh.cells = {(0, 0): cell_a, (1, 0): cell_b}
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())
    mock_mesh.min_z = 0.5
    mock_mesh.max_z = 1.5

    response = client.get("/api/malla")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    metadata = data["metadata"]
    assert metadata["radius"] == mock_mesh.radius
    assert metadata["num_cells"] == 2
    assert metadata["periodic_z"] == mock_mesh.periodic_z
    assert metadata["z_bounds"] == {"min": 0.5, "max": 1.5}
    cells_data = data["cells"]
    assert len(cells_data) == 2
    assert any(
        cd["axial_coords"]["q"] == 0 and cd["amplitude"] == 1
        for cd in cells_data)
    assert any(
        cd["axial_coords"]["q"] == 1 and cd["amplitude"] == 2
        for cd in cells_data)
    for cell_data_item in cells_data:
        assert "q_vector" in cell_data_item
        assert isinstance(cell_data_item["q_vector"], list)


def test_api_config(client, reset_globals):
    """
    Test: /api/config devuelve
    la configuración actual del módulo.
    """
    mock_mesh, mock_resonador, mock_electron, _, _ = reset_globals
    expected_malla_config_values = {
        "MW_RADIUS": "7.7",
        "MW_HEIGHT_SEG": "2",
        "MW_CIRCUM_SEG": "11",
        "MW_HEX_SIZE": "0.8",
        "MW_PERIODIC_Z": "False",
    }

    original_os_environ_get = os.environ.get

    def mock_environ_get_for_config(key, default=None):
        if key in expected_malla_config_values:
            return expected_malla_config_values[key]
        return original_os_environ_get(key, default)

    with patch("os.environ.get", side_effect=mock_environ_get_for_config):
        from watchers.watchers_tools.malla_watcher.malla_watcher import (
            MATRIZ_ECU_BASE_URL,
            TORUS_NUM_CAPAS,
            TORUS_NUM_FILAS,
            TORUS_NUM_COLUMNAS,
            AMPLITUDE_INFLUENCE_THRESHOLD,
            MAX_AMPLITUDE_FOR_NORMALIZATION,
            SIMULATION_INTERVAL,
            BASE_COUPLING_T,
            BASE_DAMPING_E,
            K_GAIN_COUPLING,
            K_GAIN_DAMPING,
            DPHI_DT_INFLUENCE_THRESHOLD,
        )

        mock_resonador.C = 0.88
        mock_electron.D = 0.12

        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        config_data = data["config"]
        m_cfg = config_data["malla_config"]
        assert m_cfg["radius"] == float(expected_malla_config_values["MW_RADIUS"])  # noqa: E501
        assert m_cfg["height_segments"] == int(expected_malla_config_values["MW_HEIGHT_SEG"])  # noqa: E501
        assert m_cfg["circumference_segments_target"] == int(expected_malla_config_values["MW_CIRCUM_SEG"])  # noqa: E501
        assert m_cfg["circumference_segments_actual"] == mock_mesh.circumference_segments_actual  # noqa: E501
        assert m_cfg["hex_size"] == float(expected_malla_config_values["MW_HEX_SIZE"])  # noqa: E501
        assert m_cfg["periodic_z"] == (expected_malla_config_values["MW_PERIODIC_Z"].lower() == "true")  # noqa: E501

        comm_cfg = config_data["communication_config"]
        assert comm_cfg["matriz_ecu_url"] == MATRIZ_ECU_BASE_URL
        torus_dims_expected = (
            f"{TORUS_NUM_CAPAS}x{TORUS_NUM_FILAS}x{TORUS_NUM_COLUMNAS}")
        assert comm_cfg["torus_dims"] == torus_dims_expected
        assert comm_cfg["influence_threshold"] == AMPLITUDE_INFLUENCE_THRESHOLD  # noqa: E501
        assert comm_cfg["max_activity_normalization"] == MAX_AMPLITUDE_FOR_NORMALIZATION  # noqa: E501

        sim_cfg = config_data["simulation_config"]
        assert sim_cfg["interval"] == SIMULATION_INTERVAL
        assert sim_cfg["dphi_dt_influence_threshold"] == DPHI_DT_INFLUENCE_THRESHOLD  # noqa: E501

        ctrl_cfg = config_data["control_config"]
        assert ctrl_cfg["base_coupling_t"] == BASE_COUPLING_T
        assert ctrl_cfg["base_damping_e"] == BASE_DAMPING_E
        assert ctrl_cfg["k_gain_coupling"] == K_GAIN_COUPLING
        assert ctrl_cfg["k_gain_damping"] == K_GAIN_DAMPING
        assert ctrl_cfg["current_coupling_C"] == pytest.approx(0.88)
        assert ctrl_cfg["current_damping_D"] == pytest.approx(0.12)


def test_api_malla_influence_push(client, reset_globals):
    """
    Test: /api/malla/influence (push)
    aplica campo vectorial externo.
    """
    mock_mesh, _, _, _, _ = reset_globals
    if not mock_mesh.cells:
        mock_mesh.cells[(0, 0)] = Cell(
            cyl_radius=1.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0, r_axial=0)

    test_field_vector_payload = [
        [[[1.1, 2.2]], [[3.3, 4.4]]],
        [[[5.5, 6.6]], [[7.7, 8.8]]],
    ]

    with patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".apply_external_field_to_mesh"
    ) as mock_apply_func:
        response = client.post(
            "/api/malla/influence",
            json={"field_vector": test_field_vector_payload}
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert "Campo vectorial externo (push) aplicado" in data["message"]
        mock_apply_func.assert_called_once_with(
            mock_mesh, test_field_vector_payload)


def test_api_malla_influence_push_invalid_payload(client, reset_globals):
    """
    Test: /api/malla/influence (push)
    maneja payloads inválidos.
    """
    reset_globals
    response_missing_key = client.post(
        "/api/malla/influence", json={"otro_dato": "valor"})
    assert response_missing_key.status_code == 400
    data_missing = response_missing_key.get_json()
    assert data_missing["status"] == "error"
    assert "falta 'field_vector'" in data_missing["message"].lower()

    response_empty_json = client.post("/api/malla/influence", json={})
    assert response_empty_json.status_code == 400
    data_empty = response_empty_json.get_json()
    assert data_empty["status"] == "error"
    assert "falta 'field_vector'" in data_empty["message"].lower()
# -- END OF FILE test_malla_watcher.py (REFINADO para Osciladores Acoplados) --
