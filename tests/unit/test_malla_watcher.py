# --- START OF FILE test_malla_watcher.py (REFINADO para Osciladores Acoplados) ---

import pytest
import sys
import os
import numpy as np
import math # Importar math para sqrt
from collections import Counter
from unittest.mock import patch, MagicMock, ANY # Importar mocking tools
import requests # Importar requests para las fixtures de mock

# --- Configuración de Ruta (Manteniendo tu ajuste) ---
# Obtiene la ruta absoluta al directorio raíz del proyecto
# Asume que este archivo está en mi-proyecto/tests/unit/
#PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# Insertar el directorio que contiene el paquete 'watchers'
# Asumiendo que malla_watcher está en watchers/watchers_tools/malla_watcher
#WATCHERS_TOOLS_DIR = os.path.join(PROJECT_ROOT, 'watchers', 'watchers_tools')
#if WATCHERS_TOOLS_DIR not in sys.path:
#    sys.path.insert(0, WATCHERS_TOOLS_DIR)

# --- Importaciones del Módulo Bajo Prueba ---
# Importar directamente desde el módulo cilindro_grafenal
from watchers.watchers_tools.malla_watcher.utils.cilindro_grafenal import (
    HexCylindricalMesh, Cell
)
# Importar directamente desde el módulo malla_watcher
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
    apply_external_field_to_mesh
)

# Configurar logging para pruebas (opcional, útil para depurar)
import logging
logging.basicConfig(level=logging.INFO) # O DEBUG para más detalle
logger = logging.getLogger(__name__)

@pytest.fixture
def client():
    """Fixture: Cliente de prueba para la aplicación Flask."""
    # Configurar la aplicación para pruebas
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- fixture para los mock requests ---
@pytest.fixture
def mock_requests_get():
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.requests.get') as mock_get:
        yield mock_get

@pytest.fixture
def mock_requests_post():
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.requests.post') as mock_post:
        yield mock_post

@pytest.fixture
def malla_para_test_aplicar_campo():
    # Usar parámetros que generen algunas celdas
    return HexCylindricalMesh(radius=3.0, height_segments=1, circumference_segments_target=6, hex_size=1.0)

# --- fixture para reseteo globales
@pytest.fixture
def reset_globals():
    """Fixture: Mocks global state for API tests."""
    # Patch ALL relevant globals
    patcher_mesh = patch('watchers.watchers_tools.malla_watcher.malla_watcher.malla_cilindrica_global')
    patcher_resonador = patch('watchers.watchers_tools.malla_watcher.malla_watcher.resonador_global')
    patcher_electron = patch('watchers.watchers_tools.malla_watcher.malla_watcher.electron_global')
    patcher_agg_state = patch('watchers.watchers_tools.malla_watcher.malla_watcher.aggregate_state', new_callable=dict)
    patcher_agg_lock = patch('watchers.watchers_tools.malla_watcher.malla_watcher.aggregate_state_lock')
    patcher_ctrl_params = patch('watchers.watchers_tools.malla_watcher.malla_watcher.control_params', new_callable=dict)
    patcher_ctrl_lock = patch('watchers.watchers_tools.malla_watcher.malla_watcher.control_lock')

    # Patch constants used in endpoints if they are read directly from module level
    # (e.g., MATRIZ_ECU_BASE_URL, TORUS_NUM_FILAS, etc.)
    # It's better to patch the constants themselves if endpoints read them directly
    # rather than relying on the real imported values.
    patchers_const = {
        'MATRIZ_ECU_BASE_URL': patch('watchers.watchers_tools.malla_watcher.malla_watcher.MATRIZ_ECU_BASE_URL', 'http://mock-ecu:8000'),
        'TORUS_NUM_CAPAS': patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_CAPAS', 3),
        'TORUS_NUM_FILAS': patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_FILAS', 4),
        'TORUS_NUM_COLUMNAS': patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_COLUMNAS', 5),
        'AMPLITUDE_INFLUENCE_THRESHOLD': patch('watchers.watchers_tools.malla_watcher.malla_watcher.AMPLITUDE_INFLUENCE_THRESHOLD', 5.0),
        'MAX_AMPLITUDE_FOR_NORMALIZATION': patch('watchers.watchers_tools.malla_watcher.malla_watcher.MAX_AMPLITUDE_FOR_NORMALIZATION', 20.0),
        'SIMULATION_INTERVAL': patch('watchers.watchers_tools.malla_watcher.malla_watcher.SIMULATION_INTERVAL', 0.5),
        'DPHI_DT_INFLUENCE_THRESHOLD': patch('watchers.watchers_tools.malla_watcher.malla_watcher.DPHI_DT_INFLUENCE_THRESHOLD', 1.0),
        'BASE_COUPLING_T': patch('watchers.watchers_tools.malla_watcher.malla_watcher.BASE_COUPLING_T', 0.6),
        'BASE_DAMPING_E': patch('watchers.watchers_tools.malla_watcher.malla_watcher.BASE_DAMPING_E', 0.1),
        'K_GAIN_COUPLING': patch('watchers.watchers_tools.malla_watcher.malla_watcher.K_GAIN_COUPLING', 0.1),
        'K_GAIN_DAMPING': patch('watchers.watchers_tools.malla_watcher.malla_watcher.K_GAIN_DAMPING', 0.05),
        'REQUESTS_TIMEOUT': patch('watchers.watchers_tools.malla_watcher.malla_watcher.REQUESTS_TIMEOUT', 2.0),
        # Constantes para malla_config que el endpoint lee de os.environ
        # Mockear las constantes del MÓDULO a los valores que ESPERAMOS que el endpoint devuelva
        # DESPUÉS de mockear os.environ.get en el test específico.
        'MESH_RADIUS': patch('watchers.watchers_tools.malla_watcher.malla_watcher.MESH_RADIUS', 7.7), # Ejemplo de valor de test
        'MESH_HEIGHT_SEGMENTS': patch('watchers.watchers_tools.malla_watcher.malla_watcher.MESH_HEIGHT_SEGMENTS', 2), # Ejemplo
        'MESH_CIRCUMFERENCE_SEGMENTS': patch('watchers.watchers_tools.malla_watcher.malla_watcher.MESH_CIRCUMFERENCE_SEGMENTS', 11), # Ejemplo
        'MESH_HEX_SIZE': patch('watchers.watchers_tools.malla_watcher.malla_watcher.MESH_HEX_SIZE', 0.8), # Ejemplo
        'MESH_PERIODIC_Z': patch('watchers.watchers_tools.malla_watcher.malla_watcher.MESH_PERIODIC_Z', False) # Ejemplo
    }
    started_patchers_const = {name: p.start() for name, p in patchers_const.items()}

    mock_mesh = patcher_mesh.start()
    mock_resonador = patcher_resonador.start()
    mock_electron = patcher_electron.start()
    mock_agg_state = patcher_agg_state.start()
    patcher_agg_lock.start() # No necesitamos el mock devuelto
    mock_ctrl_params = patcher_ctrl_params.start()
    patcher_ctrl_lock.start() # No necesitamos el mock devuelto

    # Configure mock_mesh
    # Usar la clase Cell importada de cilindro_grafenal
    mock_mesh.cells = {
        (0,0): Cell(cyl_radius=5.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0, r_axial=0),
        (1,0): Cell(cyl_radius=5.0, cyl_theta=0.5, cyl_z=0.0, q_axial=1, r_axial=0),
        (0,1): Cell(cyl_radius=5.0, cyl_theta=0.0, cyl_z=1.0, q_axial=0, r_axial=1)
    }
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())
    mock_mesh.get_cell.side_effect = lambda q, r: mock_mesh.cells.get((q, r))
    mock_mesh.radius = 5.0
    mock_mesh.height_segments = 3 # Este valor es del mock, no necesariamente de una instancia real
    mock_mesh.circumference_segments_actual = 6 # Idem
    mock_mesh.hex_size = 1.0 # Idem
    mock_mesh.periodic_z = False # Idem
    mock_mesh.min_z = 0.0 # Idem
    mock_mesh.max_z = 1.0 # Idem
    # mock_mesh._calculate_z_bounds.return_value = None # _calculate_z_bounds no existe en la interfaz pública
    mock_mesh.verify_connectivity.return_value = {6: len(mock_mesh.cells)}
    mock_mesh.previous_flux = 0.0

    # Configure mock_resonador and mock_electron
    mock_resonador.C = started_patchers_const['BASE_COUPLING_T'] # Usar el valor mockeado de la constante
    mock_electron.D = started_patchers_const['BASE_DAMPING_E'] # Usar el valor mockeado
    mock_resonador.ajustar_coeficientes.return_value = None
    mock_electron.ajustar_coeficientes.return_value = None

    # Configure mock_agg_state
    mock_agg_state.update({
        "avg_amplitude": 0.0, "max_amplitude": 0.0, "avg_velocity": 0.0, "max_velocity": 0.0,
        "avg_kinetic_energy": 0.0, "max_kinetic_energy": 0.0,
        "avg_activity_magnitude": 0.0, "max_activity_magnitude": 0.0, "cells_over_threshold": 0
    })

    # Configure mock_ctrl_params
    mock_ctrl_params.update({
        "phoswave_C": started_patchers_const['BASE_COUPLING_T'],
        "electron_D": started_patchers_const['BASE_DAMPING_E']
    })

    # No necesitamos mock_get y mock_post aquí si usamos las fixtures dedicadas
    yield mock_mesh, mock_resonador, mock_electron, mock_agg_state, mock_ctrl_params # No devolver mock_get, mock_post

    patcher_mesh.stop()
    patcher_resonador.stop()
    patcher_electron.stop()
    patcher_agg_state.stop()
    patcher_agg_lock.stop()
    patcher_ctrl_params.stop()
    patcher_ctrl_lock.stop()
    for p in patchers_const.values():
        p.stop()

# --- Tests para la Clase PhosWave ---
def test_phoswave_initialization():
    """Test: Inicialización correcta de atributos de PhosWave."""
    wave = PhosWave(coef_acoplamiento=0.7) # Nuevo constructor
    assert wave.C == 0.7
    # Asegurar que el coeficiente negativo se ajusta a 0
    wave_neg = PhosWave(coef_acoplamiento=-0.5)
    assert wave_neg.C == 0.0

def test_phoswave_ajustar_coeficientes():
    """Test: Ajuste correcto del coeficiente de acoplamiento."""
    wave = PhosWave(coef_acoplamiento=0.7)
    wave.ajustar_coeficientes(0.9)
    assert wave.C == 0.9
    # Ajuste negativo
    wave.ajustar_coeficientes(-0.2)
    assert wave.C == 0.0

# --- Tests para la Clase Electron ---
def test_electron_initialization():
    """Test: Inicialización correcta de atributos de Electron."""
    elec = Electron(coef_amortiguacion=0.3) # Nuevo constructor
    assert elec.D == 0.3
    # Asegurar que el coeficiente negativo se ajusta a 0
    elec_neg = Electron(coef_amortiguacion=-0.1)
    assert elec_neg.D == 0.0

def test_electron_ajustar_coeficientes():
    """Test: Ajuste correcto del coeficiente de amortiguación."""
    elec = Electron(coef_amortiguacion=0.3)
    elec.ajustar_coeficientes(0.5)
    assert elec.D == 0.5
    # Ajuste negativo
    elec.ajustar_coeficientes(-0.05)
    assert elec.D == 0.0

def test_apply_external_field_to_mesh_logic(malla_para_test_aplicar_campo): # Correcto
    """Test: Aplicación de un campo vectorial externo a q_vector usando interpolación."""
    from watchers.watchers_tools.malla_watcher.malla_watcher import apply_external_field_to_mesh # Importar aquí
    
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

    apply_external_field_to_mesh(mesh_instance, external_field_list)

    changed_count = 0
    for coords, cell in mesh_instance.cells.items():
        assert isinstance(cell.q_vector, np.ndarray)
        assert cell.q_vector.shape == (2,)
        # Esta aserción es más robusta si la interpolación puede dar cero
        is_zero_vector = np.array_equal(cell.q_vector, np.zeros(2))
        if not is_zero_vector:
             changed_count +=1
        # Opcional: si esperas que todos cambien y no sean cero:
        # assert not np.array_equal(cell.q_vector, np.zeros(2)), f"q_vector para {coords} no cambió de cero"
    
    # Si la malla es muy pequeña o la interpolación da cero para todas las celdas, changed_count puede ser 0.
    # La aserción debe ser más específica o la prueba debe garantizar que algunos q_vectors cambien.
    # Por ahora, asumimos que al menos uno debería cambiar si la malla no es trivial.
    if len(mesh_instance.cells) > 0 : # Solo si hay celdas
        assert changed_count > 0 or any(np.linalg.norm(c.q_vector) > 1e-9 for c in mesh_instance.cells.values()), \
            "apply_external_field_to_mesh no modificó ningún q_vector de forma no nula."
    logger.info(f"{changed_count} de {len(mesh_instance.cells)} celdas actualizaron su q_vector a no-cero.")

# --- Tests para la Lógica de Simulación ---

@pytest.fixture
def mock_malla_sim():
    """Fixture: Configura una pequeña malla mockeada para simulación."""
    # Crear celdas manualmente para un escenario simple
    # Inicializar con q_vector
    cell_center = Cell(5.0, 0.0, 0.0, 0, 0, amplitude=10.0, velocity=0.0, q_vector=np.array([0.1, 0.2]))
    cell_neighbor1 = Cell(5.0, 0.5, 0.0, 1, 0, amplitude=0.0, velocity=0.0, q_vector=np.array([-0.1, -0.2]))
    cell_neighbor2 = Cell(5.0, -0.5, 0.0, -1, 0, amplitude=0.0, velocity=0.0, q_vector=np.array([0.0, 0.0]))

    # Mockear la instancia global de la malla
    mock_mesh = MagicMock(spec=HexCylindricalMesh)
    mock_mesh.cells = {(0,0): cell_center, (1,0): cell_neighbor1, (-1,0): cell_neighbor2}
    mock_mesh.get_all_cells.return_value = [cell_center, cell_neighbor1, cell_neighbor2]

    # Configurar get_neighbor_cells para este escenario simple
    def mock_get_neighbors(q, r):
        if (q, r) == (0, 0):
            return [cell_neighbor1, cell_neighbor2]
        elif (q, r) == (1, 0):
            return [cell_center] # Simplificado: solo el centro como vecino
        elif (q, r) == (-1, 0):
            return [cell_center] # Simplificado
        return []

    mock_mesh.get_neighbor_cells.side_effect = mock_get_neighbors
    mock_mesh.get_cell.side_effect = lambda q, r: mock_mesh.cells.get((q, r))
    # Mockear previous_flux en la malla mockeada
    mock_mesh.previous_flux = 0.0

    # Mockear las instancias globales de PhosWave y Electron
    mock_resonador = MagicMock(spec=PhosWave)
    mock_resonador.C = 0.5 # Coeficiente de acoplamiento de prueba

    mock_electron = MagicMock(spec=Electron)
    mock_electron.D = 0.1 # Coeficiente de amortiguación de prueba

    # Mockear el intervalo de simulación
    mock_sim_interval = 0.1

    # Usar patch para reemplazar las instancias globales y la constante
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.malla_cilindrica_global', mock_mesh), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.resonador_global', mock_resonador), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.electron_global', mock_electron), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.SIMULATION_INTERVAL', mock_sim_interval):
        # Yield mocks if tests need them directly
        yield cell_center, cell_neighbor1, cell_neighbor2, mock_mesh, mock_resonador, mock_electron, mock_sim_interval

def test_simular_paso_malla_propagation(mock_malla_sim):
    """Test: simular_paso_malla propaga amplitud/velocidad correctamente (acoplamiento)."""
    cell_center, cell_neighbor1, cell_neighbor2, mock_mesh, mock_resonador, mock_electron, dt = mock_malla_sim

    # Configurar estado inicial para ver propagación
    cell_center.amplitude = 10.0
    cell_center.velocity = 0.0
    cell_center.q_vector = np.array([0.1, 0.2])
    cell_neighbor1.amplitude = 0.0
    cell_neighbor1.velocity = 0.0
    cell_neighbor1.q_vector = np.array([-0.1, -0.2])
    cell_neighbor2.amplitude = 0.0
    cell_neighbor2.velocity = 0.0
    cell_neighbor2.q_vector = np.array([0.0, 0.0])

    mock_resonador.C = 0.5 # Coeficiente de acoplamiento base
    mock_electron.D = 0.0 # Sin amortiguación por ahora

    # Calcular valores esperados después de un paso (usando cálculos más precisos)
    # Celda Central (0,0):
    norm_q_center = np.linalg.norm(cell_center.q_vector) # sqrt(0.05)
    modulated_C_center = mock_resonador.C * max(0.0, 1.0 + norm_q_center) # 0.5 * (1 + sqrt(0.05))
    # Force on center from n1: modulated_C_center * (amp_n1 - amp_c)
    # Force on center from n2: modulated_C_center * (amp_n2 - amp_c)
    F_coupling_c = modulated_C_center * (cell_neighbor1.amplitude - cell_center.amplitude) + \
                   modulated_C_center * (cell_neighbor2.amplitude - cell_center.amplitude)
    # F_coupling_c = modulated_C_center * (0 - 10) + modulated_C_center * (0 - 10) = -20 * modulated_C_center
    # Accel_c = -20 * modulated_C_center
    # next_vel_c = 0 + Accel_c * dt
    expected_next_vel_c = -20 * modulated_C_center * dt
    expected_next_amp_c = cell_center.amplitude + expected_next_vel_c * dt

    # Celda Vecina 1 (1,0):
    norm_q_n1 = np.linalg.norm(cell_neighbor1.q_vector) # sqrt(0.05)
    modulated_C_n1 = mock_resonador.C * max(0.0, 1.0 + norm_q_n1) # 0.5 * (1 + sqrt(0.05))
    # Force on n1 from center: modulated_C_n1 * (amp_c - amp_n1)
    F_coupling_n1 = modulated_C_n1 * (cell_center.amplitude - cell_neighbor1.amplitude)
    # Accel_n1 = F_coupling_n1
    # next_vel_n1 = 0 + Accel_n1 * dt
    expected_next_vel_n1 = F_coupling_n1 * dt
    expected_next_amp_n1 = cell_neighbor1.amplitude + expected_next_vel_n1 * dt

    # Celda Vecina 2 (-1,0):
    norm_q_n2 = np.linalg.norm(cell_neighbor2.q_vector) # 0.0
    modulated_C_n2 = mock_resonador.C * max(0.0, 1.0 + norm_q_n2) # 0.5 * (1 + 0) = 0.5
    # Force on n2 from center: modulated_C_n2 * (amp_c - amp_n2)
    F_coupling_n2 = modulated_C_n2 * (cell_center.amplitude - cell_neighbor2.amplitude)
    # Accel_n2 = F_coupling_n2
    # next_vel_n2 = 0 + Accel_n2 * dt
    expected_next_vel_n2 = F_coupling_n2 * dt
    expected_next_amp_n2 = cell_neighbor2.amplitude + expected_next_vel_n2 * dt

    # Acción
    simular_paso_malla()

    # Verificación
    assert cell_center.amplitude == pytest.approx(expected_next_amp_c)
    assert cell_center.velocity == pytest.approx(expected_next_vel_c)
    assert cell_neighbor1.amplitude == pytest.approx(expected_next_amp_n1)
    assert cell_neighbor1.velocity == pytest.approx(expected_next_vel_n1)
    assert cell_neighbor2.amplitude == pytest.approx(expected_next_amp_n2)
    assert cell_neighbor2.velocity == pytest.approx(expected_next_vel_n2)

def test_simular_paso_malla_damping(mock_malla_sim):
    """Test: simular_paso_malla aplica amortiguación correctamente."""
    cell_center, cell_neighbor1, cell_neighbor2, mock_mesh, mock_resonador, mock_electron, dt = mock_malla_sim

    # Configurar estado inicial con velocidad y sin acoplamiento
    cell_center.amplitude = 10.0
    cell_center.velocity = 5.0
    cell_center.q_vector = np.zeros(2) # Sin q_vector para simplificar modulación
    cell_neighbor1.amplitude = 0.0
    cell_neighbor1.velocity = 0.0
    cell_neighbor1.q_vector = np.zeros(2)
    cell_neighbor2.amplitude = 0.0
    cell_neighbor2.velocity = 0.0
    cell_neighbor2.q_vector = np.zeros(2)

    mock_resonador.C = 0.0 # Sin acoplamiento
    mock_electron.D = 0.2 # Coeficiente de amortiguación de prueba

    # Calcular valores esperados para Celda Central (sin acoplamiento, solo amortiguación)
    # F_coupling_c = 0.0
    # F_damping_c = - D * vel_c = - 0.2 * 5.0 = -1.0
    # Net Force = -1.0
    # Accel = -1.0
    # next_vel_c = vel_c + Accel * dt = 5.0 + (-1.0) * 0.1 = 5.0 - 0.1 = 4.9
    # next_amp_c = amp_c + next_vel_c * dt = 10.0 + 4.9 * 0.1 = 10.0 + 0.49 = 10.49
    expected_next_vel_c = 5.0 + (-mock_electron.D * 5.0) * dt
    expected_next_amp_c = 10.0 + expected_next_vel_c * dt

    # Vecinas no cambian (sin acoplamiento ni velocidad inicial)
    expected_next_amp_n1 = 0.0
    expected_next_vel_n1 = 0.0
    expected_next_amp_n2 = 0.0
    expected_next_vel_n2 = 0.0

    # Acción
    simular_paso_malla()

    # Verificación
    assert cell_center.amplitude == pytest.approx(expected_next_amp_c)
    assert cell_center.velocity == pytest.approx(expected_next_vel_c)
    assert cell_neighbor1.amplitude == pytest.approx(expected_next_amp_n1)
    assert cell_neighbor1.velocity == pytest.approx(expected_next_vel_n1)
    assert cell_neighbor2.amplitude == pytest.approx(expected_next_amp_n2)
    assert cell_neighbor2.velocity == pytest.approx(expected_next_vel_n2)

def test_calculate_flux():
    """Test: calculate_flux suma la componente vy de q_vector."""
    mesh = HexCylindricalMesh(radius=1.0, height_segments=1, circumference_segments_target=3, hex_size=1.0)
    mesh.cells.clear()
    # Usar los nombres de parámetros correctos para Cell
    cell1 = Cell(cyl_radius=1.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0, r_axial=0, q_vector=np.array([1.0, 2.0]))
    cell2 = Cell(cyl_radius=1.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=1, r_axial=0, q_vector=np.array([-0.5, 3.0]))
    cell3 = Cell(cyl_radius=1.0, cyl_theta=0.0, cyl_z=1.0, q_axial=0, r_axial=1, q_vector=np.array([0.0, -1.5]))
    mesh.cells[(0,0)] = cell1
    mesh.cells[(1,0)] = cell2
    mesh.cells[(0,1)] = cell3
    expected_flux = 2.0 + 3.0 + (-1.5)
    actual_flux = calculate_flux(mesh)
    assert actual_flux == pytest.approx(expected_flux)
    mesh_empty = HexCylindricalMesh(radius=1.0, height_segments=1, circumference_segments_target=3, hex_size=1.0)
    mesh_empty.cells.clear()
    assert calculate_flux(mesh_empty) == pytest.approx(0.0)

# NUEVO TEST: dPhi/dt calculation
def test_dphi_dt_calculation(mock_malla_sim):
    """Test: Cálculo de dPhi/dt (esto es inducción electromagnética) en simulation_loop."""
    cell_center, cell_neighbor1, cell_neighbor2, mock_mesh, mock_resonador, mock_electron, dt = mock_malla_sim

    # Mockear calculate_flux para retornar valores predefinidos
    patcher_calculate_flux = patch('watchers.watchers_tools.malla_watcher.malla_watcher.calculate_flux')
    mock_calculate_flux = patcher_calculate_flux.start()

    # Configurar previous_flux inicial en la malla mockeada
    mock_mesh.previous_flux = 10.0
    initial_previous_flux = mock_mesh.previous_flux

    # Configurar calculate_flux para retornar un valor en la primera llamada
    mock_calculate_flux.return_value = 15.0
    current_flux_step1 = mock_calculate_flux.return_value

    # Simular la parte relevante de simulation_loop (cálculo de dPhi/dt)
    # Esto requiere mockear o simular el bucle de simulación
    # Una forma más simple es testear la lógica de cálculo de dPhi/dt directamente si está en una función separada,
    # o mockear las dependencias de simulation_loop (fetch_and_apply_torus_field, simular_paso_malla, update_aggregate_state)
    # y ejecutar un paso del loop.

    # Vamos a mockear las funciones llamadas dentro del loop y ejecutar un paso
    patcher_fetch = patch('watchers.watchers_tools.malla_watcher.malla_watcher.fetch_and_apply_torus_field')
    patcher_sim_paso = patch('watchers.watchers_tools.malla_watcher.malla_watcher.simular_paso_malla')
    patcher_update_state = patch('watchers.watchers_tools.malla_watcher.malla_watcher.update_aggregate_state')
    patcher_send_influence = patch('watchers.watchers_tools.malla_watcher.malla_watcher.send_influence_to_torus')
    patcher_stop_event = patch('watchers.watchers_tools.malla_watcher.malla_watcher.stop_simulation_event')

    mock_fetch = patcher_fetch.start()
    mock_sim_paso = patcher_sim_paso.start()
    mock_update_state = patcher_update_state.start()
    mock_send_influence = patcher_send_influence.start()
    mock_stop_event = patcher_stop_event.start()

    # Configurar stop_event para que el loop se ejecute solo una vez
    mock_stop_event.is_set.side_effect = [False, True] # Primera llamada False, segunda True
    mock_stop_event.wait.return_value = None # No esperar

    # Configurar calculate_flux para retornar valores para dos pasos
    mock_calculate_flux.side_effect = [15.0, 18.0] # Flujo en paso 1, Flujo en paso 2

    # Ejecutar el bucle de simulación (que ahora solo corre una iteración)
    # Necesitamos llamar a simulation_loop()
    # Asegurarse de que SIMULATION_INTERVAL está mockeado a dt
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.SIMULATION_INTERVAL', dt):
         simulation_loop() # Ejecutar una iteración

    # Verificación del cálculo de dPhi/dt en la primera iteración
    # dPhi/dt = (current_flux - previous_flux) / dt
    # dPhi/dt = (15.0 - 10.0) / 0.1 = 5.0 / 0.1 = 50.0
    expected_dphi_dt_step1 = (15.0 - initial_previous_flux) / dt
    # Verificar que previous_flux se actualizó
    assert mock_mesh.previous_flux == pytest.approx(15.0)

    # Verificar que calculate_flux fue llamado
    mock_calculate_flux.assert_called_once_with(mock_mesh)

    # Verificar que send_influence_to_torus fue llamado con el dPhi/dt calculado (si supera el umbral)
    # Asumimos DPHI_DT_INFLUENCE_THRESHOLD = 1.0 por defecto. 50.0 > 1.0
    expected_dphi_dt_sent = expected_dphi_dt_step1
    # CORREGIDO: Verificar que send_influence_to_torus fue llamado con el valor esperado
    mock_send_influence.assert_called_once_with(pytest.approx(expected_dphi_dt_sent))

    # Limpiar mocks
    patcher_calculate_flux.stop()
    patcher_fetch.stop()
    patcher_sim_paso.stop()
    patcher_update_state.stop()
    patcher_send_influence.stop()
    patcher_stop_event.stop()

# NUEVO TEST: send_influence_to_torus
def test_send_influence_to_torus(mock_requests_post):
    """Test: send_influence_to_torus envía POST a ECU con dPhi/dt."""
    mock_post = mock_requests_post # Usar la fixture

    dphi_dt_value = 7.5
    # Mockear las constantes TORUS_NUM_FILAS y TORUS_NUM_COLUMNAS para este test
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_FILAS', 10), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_COLUMNAS', 15), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.MATRIZ_ECU_BASE_URL', 'http://mock-ecu:8000'), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.REQUESTS_TIMEOUT', 2.0):
        
        expected_target_capa = 0
        expected_target_row = 10 // 2
        expected_target_col = 15 // 2
        
        send_influence_to_torus(dphi_dt_value)

        mock_post.assert_called_once_with(
            'http://mock-ecu:8000/api/ecu/influence',
            json={
                'capa': expected_target_capa,
                'row': expected_target_row,
                'col': expected_target_col,
                'vector': [dphi_dt_value, 0.0],
                'nombre_watcher': f'malla_watcher_dPhiDt{dphi_dt_value:.3f}'
            },
            timeout=pytest.approx(2.0)
        )

# --- Tests de Comunicación ---

# NUEVO TEST: fetch_and_apply_torus_field
def test_fetch_and_apply_torus_field(mock_requests_get): # mock_requests_get es la fixture
    """Test: fetch_and_apply_torus_field obtiene campo vectorial y llama a apply_external_field_to_mesh."""
    mock_get = mock_requests_get # Usar la fixture

    mock_ecu_field_data = [
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], # Ejemplo más completo
        [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]]
    ]
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_ecu_field_data
    mock_get.return_value.raise_for_status.return_value = None

    # Mockear las dependencias globales que usa fetch_and_apply_torus_field
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.malla_cilindrica_global') as mock_mesh_global_instance, \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.apply_external_field_to_mesh') as mock_apply_func, \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.MATRIZ_ECU_BASE_URL', 'http://mock-ecu:8000'), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.REQUESTS_TIMEOUT', 3.0):

        # Configurar el mock_mesh_global_instance para que no sea None y tenga 'cells'
        # Es importante que sea un objeto que pase la condición `if mesh:` en fetch_and_apply_torus_field
        mock_mesh_global_instance.configure_mock(cells={ (0,0): "dummy_cell" }) # Solo necesita ser no-None y tener .cells

        # No es necesario re-importar fetch_and_apply_torus_field aquí si ya está importada al inicio del archivo
        fetch_and_apply_torus_field() # Llamar a la función importada

        mock_get.assert_called_once_with('http://mock-ecu:8000/api/ecu/field_vector', timeout=pytest.approx(3.0))
        mock_get.return_value.raise_for_status.assert_called_once()
        mock_apply_func.assert_called_once_with(mock_mesh_global_instance, mock_ecu_field_data)

# --- Tests para Estado Agregado (NUEVOS) ---

@pytest.fixture
def mock_malla_state():
    """Fixture: Configura una malla mockeada con amplitudes para estado agregado."""
    # Inicializar con q_vector
    cell1 = Cell(5.0, 0.0, 0.0, 0, 0, amplitude=10.0, velocity=1.0, q_vector=np.array([0.1, 0.2]))
    cell2 = Cell(5.0, 0.5, 0.0, 1, 0, amplitude=-5.0, velocity=-2.0, q_vector=np.array([-0.1, -0.2])) # CORREGIDO: velocity a -2.0 para tener valores variados
    cell3 = Cell(5.0, 1.0, 0.0, 2, 0, amplitude=2.0, velocity=0.1, q_vector=np.array([0.0, 0.0]))
    cell4 = Cell(5.0, 1.5, 0.0, 3, 0, amplitude=6.0, velocity=3.0, q_vector=np.array([0.3, 0.4])) # CORREGIDO: velocity a 3.0

    mock_mesh = MagicMock(spec=HexCylindricalMesh)
    mock_mesh.cells = {(0,0): cell1, (1,0): cell2, (2,0): cell3, (3,0): cell4}
    mock_mesh.get_all_cells.return_value = [cell1, cell2, cell3, cell4]
    # Mockear previous_flux en la malla mockeada (no usado directamente en update_aggregate_state, pero buena práctica)
    mock_mesh.previous_flux = 0.0

    # Usar patch para reemplazar las instancias globales y constantes
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.malla_cilindrica_global', mock_mesh), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.AMPLITUDE_INFLUENCE_THRESHOLD', 5.0), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.DPHI_DT_INFLUENCE_THRESHOLD', 1.0), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.SIMULATION_INTERVAL', 0.5), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.aggregate_state_lock'): # Mockear el lock
        # Yield mocks if tests need them directly
        yield mock_mesh, [cell1, cell2, cell3, cell4]

def test_update_aggregate_state(mock_malla_state):
    """Test: update_aggregate_state calcula métricas basadas en amplitud, velocidad y actividad."""
    mock_mesh, cells = mock_malla_state

    # Calcular valores esperados basándose en las celdas de la fixture
    # Amplitudes: [10.0, -5.0, 2.0, 6.0]
    # Velocidades: [1.0, -2.0, 0.1, 3.0]
    # Energías Cinéticas (0.5 * v^2): [0.5*1^2=0.5, 0.5*(-2)^2=2.0, 0.5*0.1^2=0.005, 0.5*3^2=4.5]
    # Actividad (sqrt(amp^2 + vel^2)): [sqrt(10^2+1^2)=sqrt(101)~10.05, sqrt((-5)^2+(-2)^2)=sqrt(29)~5.38, sqrt(2^2+0.1^2)=sqrt(4.01)~2.00, sqrt(6^2+3^2)=sqrt(45)~6.71]

    expected_avg_amp = (10.0 - 5.0 + 2.0 + 6.0) / 4.0 # 13.0 / 4.0 = 3.25
    expected_max_amp = 10.0
    expected_avg_vel = (1.0 - 2.0 + 0.1 + 3.0) / 4.0 # 2.1 / 4.0 = 0.525
    expected_max_vel = 3.0
    expected_avg_ke = (0.5 + 2.0 + 0.005 + 4.5) / 4.0 # 7.005 / 4.0 = 1.75125
    expected_max_ke = 4.5
    expected_activity_mags = [math.sqrt(101), math.sqrt(29), math.sqrt(4.01), math.sqrt(45)]
    expected_avg_activity = sum(expected_activity_mags) / 4.0
    expected_max_activity = max(expected_activity_mags)

    # Umbral = 5.0. Celdas con actividad > 5.0 son cell1 (~10.05), cell2 (~5.38), cell4 (~6.71)
    expected_over_thresh = 3

    # Acción
    update_aggregate_state()

    # Verificación
    from watchers.watchers_tools.malla_watcher.malla_watcher import aggregate_state
    assert aggregate_state["avg_amplitude"] == pytest.approx(expected_avg_amp)
    assert aggregate_state["max_amplitude"] == pytest.approx(expected_max_amp)
    assert aggregate_state["avg_velocity"] == pytest.approx(expected_avg_vel) # NUEVO
    assert aggregate_state["max_velocity"] == pytest.approx(expected_max_vel) # NUEVO
    assert aggregate_state["avg_kinetic_energy"] == pytest.approx(expected_avg_ke) # NUEVO
    assert aggregate_state["max_kinetic_energy"] == pytest.approx(expected_max_ke) # NUEVO
    assert aggregate_state["avg_activity_magnitude"] == pytest.approx(expected_avg_activity) # NUEVO
    assert aggregate_state["max_activity_magnitude"] == pytest.approx(expected_max_activity) # NUEVO
    assert aggregate_state["cells_over_threshold"] == expected_over_thresh

    # Test case for empty mesh (should reset state)
    mock_mesh_empty = MagicMock(spec=HexCylindricalMesh)
    mock_mesh_empty.cells = {} # Empty cells dict
    mock_mesh_empty.get_all_cells.return_value = [] # get_all_cells should return empty list

    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.malla_cilindrica_global', mock_mesh_empty), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.aggregate_state_lock'):
         update_aggregate_state()
         # Verify state is reset to zeros
         assert aggregate_state["avg_amplitude"] == 0.0
         assert aggregate_state["max_amplitude"] == 0.0
         assert aggregate_state["avg_velocity"] == 0.0
         assert aggregate_state["max_velocity"] == 0.0
         assert aggregate_state["avg_kinetic_energy"] == 0.0
         assert aggregate_state["max_kinetic_energy"] == 0.0
         assert aggregate_state["avg_activity_magnitude"] == 0.0
         assert aggregate_state["max_activity_magnitude"] == 0.0
         assert aggregate_state["cells_over_threshold"] == 0

# --- Tests para Mapeo a Toroide (NUEVOS) ---

@pytest.fixture
def mock_malla_map():
    """Fixture: Configura una malla mockeada para pruebas de mapeo."""
    # Crear una malla simple con límites Z definidos
    mock_mesh = MagicMock(spec=HexCylindricalMesh)
    mock_mesh.cells = {} # No necesitamos celdas reales, solo los límites Z
    mock_mesh.min_z = -10.0
    mock_mesh.max_z = 10.0
    mock_mesh.circumference_segments_actual = 12 # Para mapeo theta
    # NUEVO: Mockear previous_flux
    mock_mesh.previous_flux = 0.0

    # Mockear constantes del toroide
    mock_torus_capas = 5
    mock_torus_filas = 10
    mock_torus_columnas = 15

    # Usar patch para reemplazar las constantes
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.malla_cilindrica_global', mock_mesh), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_CAPAS', mock_torus_capas), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_FILAS', mock_torus_filas), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_COLUMNAS', mock_torus_columnas), \
         patch('watchers.watchers_tools.malla_watcher.malla_watcher.MAX_AMPLITUDE_FOR_NORMALIZATION', 20.0):
        # Yield mocks/values if tests need them directly
        yield mock_mesh, mock_torus_capas, mock_torus_filas, mock_torus_columnas

def test_map_cylinder_to_torus_coords(mock_malla_map):
    """Test: map_cylinder_to_torus_coords mapea correctamente usando actividad."""
    mock_mesh, num_capas, num_filas, num_columnas = mock_malla_map
    
    # Usar los nombres de parámetros correctos para Cell
    cell = Cell(cyl_radius=5.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=10, r_axial=20, amplitude=10.0, velocity=1.0, q_vector=np.array([0.0, 0.0]))
    mock_mesh.cells[(-99, -99)] = Cell(cyl_radius=0.0, cyl_theta=0.0, cyl_z=0.0, q_axial=-99, r_axial=-99) # Asegurar que mock_mesh.cells no esté vacío

    expected_coords = (2, 4, 7)
    actual_coords = map_cylinder_to_torus_coords(cell)
    assert actual_coords == expected_coords

    cell_zero_act = Cell(cyl_radius=5.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=10, r_axial=20, amplitude=0.0, velocity=0.0, q_vector=np.array([0.0, 0.0]))
    expected_coords_zero = (4, 4, 7)
    actual_coords_zero = map_cylinder_to_torus_coords(cell_zero_act)
    assert actual_coords_zero == expected_coords_zero

    cell_max_act = Cell(cyl_radius=5.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=10, r_axial=20, amplitude=20.0, velocity=0.0, q_vector=np.array([0.0, 0.0]))
    expected_coords_max = (0, 4, 7)
    actual_coords_max = map_cylinder_to_torus_coords(cell_max_act)
    assert actual_coords_max == expected_coords_max

    cell_high_act = Cell(cyl_radius=5.0, cyl_theta=np.pi, cyl_z=0.0, q_axial=10, r_axial=20, amplitude=15.0, velocity=15.0, q_vector=np.array([0.0, 0.0]))
    expected_coords_high = (0, 4, 7)
    actual_coords_high = map_cylinder_to_torus_coords(cell_high_act)
    assert actual_coords_high == expected_coords_high

# --- Tests para Envío de Influencia (NUEVOS) ---

@pytest.fixture
def mock_send_influence():
     """Fixture: Configura mocks para probar send_influence_to_torus."""
     # Patch requests.post
     patcher_requests_post = patch('watchers.watchers_tools.malla_watcher.malla_watcher.requests.post')
     mock_post = patcher_requests_post.start()
     mock_post.return_value.raise_for_status.return_value = None # No lanzar excepción por defecto

     # Patch constantes usadas en send_influence_to_torus
     patcher_ecu_url = patch('watchers.watchers_tools.malla_watcher.malla_watcher.MATRIZ_ECU_BASE_URL', 'http://mock-ecu:8000')
     patcher_torus_dims = patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_FILAS', 10) # Necesario para calcular target_row/col
     patcher_torus_cols = patch('watchers.watchers_tools.malla_watcher.malla_watcher.TORUS_NUM_COLUMNAS', 15) # Necesario para calcular target_row/col
     patcher_timeout = patch('watchers.watchers_tools.malla_watcher.malla_watcher.REQUESTS_TIMEOUT', 2.0) # Mockear el timeout

     mock_ecu_url = patcher_ecu_url.start()
     mock_torus_dims = patcher_torus_dims.start()
     mock_torus_cols = patcher_torus_cols.start()
     mock_timeout = patcher_timeout.start()

     yield mock_post # Yield the mock post object

     # Clean up mocks
     patcher_requests_post.stop()
     patcher_ecu_url.stop()
     patcher_torus_dims.stop()
     patcher_torus_cols.stop()
     patcher_timeout.stop()

# --- Tests de API (NUEVOS) ---

def test_api_health(client, reset_globals):
    """Test: Endpoint /api/health retorna estado correcto y detalles de la malla."""
    mock_mesh, mock_resonador, mock_electron, mock_agg_state, mock_ctrl_params = reset_globals

    # Configurar estado específico para este test de salud
    mock_mesh.cells = {
        (0,0): Cell(cyl_radius=1.0, cyl_theta=1.0, cyl_z=1.0, q_axial=0, r_axial=0),
        (1,0): Cell(cyl_radius=1.0, cyl_theta=1.0, cyl_z=1.0, q_axial=1, r_axial=0),
        (0,1): Cell(cyl_radius=1.0, cyl_theta=1.0, cyl_z=1.0, q_axial=0, r_axial=1)
    }
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())
    # Simular una conectividad perfecta para este test
    mock_mesh.verify_connectivity.side_effect = lambda: {6: len(mock_mesh.cells)}
    # Simular que el hilo de simulación está activo
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.simulation_thread') as mock_sim_thread:
        mock_sim_thread.is_alive.return_value = True

        response = client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()

        assert data['status'] == 'success'
        assert data['module'] == 'Malla_watcher'
        assert data['message'] == 'Malla_watcher operativo.' # Mensaje esperado para éxito

        details = data['details']
        assert details['mesh']['initialized'] is True
        assert details['mesh']['num_cells'] == 3
        assert details['mesh']['connectivity_status'] == 'ok'
        assert details['mesh']['min_neighbors'] == 6
        assert details['mesh']['max_neighbors'] == 6
        # Los valores de periodic_z, min_z, max_z vienen de la configuración de mock_mesh en reset_globals
        assert details['mesh']['z_periodic'] == mock_mesh.periodic_z
        assert details['resonator_simulation']['running'] is True

def test_api_health_no_simulation_thread(client, reset_globals):
    """Test: Endpoint /api/health cuando el hilo de simulación no está activo."""
    mock_mesh, _, _, _, _ = reset_globals
    mock_mesh.cells = {(0,0): Cell(cyl_radius=1.0, cyl_theta=1.0, cyl_z=1.0, q_axial=0, r_axial=0)} # Malla no vacía
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())

    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.simulation_thread') as mock_sim_thread:
        mock_sim_thread.is_alive.return_value = False # Hilo inactivo

        response = client.get('/api/health')
        # El código de estado podría ser 200 con status "error" o 503.
        # Según la lógica actual, parece que será 500 o 503 si el status es "error".
        # Si el status es "error" por hilo inactivo, el código HTTP es 500.
        assert response.status_code == 500 # O el código que decidas para este caso
        data = response.get_json()
        assert data['status'] == 'error'
        assert "Hilo de simulación del resonador inactivo" in data['message']
        assert data['details']['resonator_simulation']['running'] is False

def test_api_health_empty_mesh(client, reset_globals):
    """Test: Endpoint /api/health cuando la malla está inicializada pero vacía."""
    mock_mesh, _, _, _, _ = reset_globals
    mock_mesh.cells = {} # Malla vacía
    mock_mesh.get_all_cells.return_value = []

    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.simulation_thread') as mock_sim_thread:
        mock_sim_thread.is_alive.return_value = True # Hilo activo

        response = client.get('/api/health')
        assert response.status_code == 500 # O el código que decidas
        data = response.get_json()
        assert data['status'] == 'error'
        assert "Malla inicializada pero contiene 0 celdas" in data['message']
        assert data['details']['mesh']['num_cells'] == 0

def test_api_state(client, reset_globals):
    """Test: Endpoint /api/state retorna el estado agregado y los parámetros de control correctos."""
    mock_mesh, mock_resonador, mock_electron, mock_agg_state, mock_ctrl_params = reset_globals

    # Configurar valores específicos para el estado agregado
    mock_agg_state.update({
        "avg_amplitude": 12.3, "max_amplitude": 45.6, "avg_velocity": 1.5, "max_velocity": 3.0,
        "avg_kinetic_energy": 0.75, "max_kinetic_energy": 4.5,
        "avg_activity_magnitude": 13.0, "max_activity_magnitude": 46.0, "cells_over_threshold": 7
    })

    # Configurar celdas específicas en el mock_mesh para este test
    mock_mesh.cells = {
        (0,0): Cell(cyl_radius=5.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0, r_axial=0),
        (1,0): Cell(cyl_radius=5.0, cyl_theta=0.5, cyl_z=0.0, q_axial=1, r_axial=0),
        (0,1): Cell(cyl_radius=5.0, cyl_theta=0.0, cyl_z=1.0, q_axial=0, r_axial=1)
    }
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())
    expected_num_cells = len(mock_mesh.cells)

    # Configurar parámetros de control específicos
    expected_C = 0.75
    expected_D = 0.25
    mock_resonador.C = expected_C
    mock_electron.D = expected_D
    mock_ctrl_params.update({"phoswave_C": expected_C, "electron_D": expected_D})

    response = client.get('/api/state')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    state_data = data['state']

    assert state_data['avg_amplitude'] == pytest.approx(12.3)
    assert state_data['max_amplitude'] == pytest.approx(45.6)
    assert state_data['avg_velocity'] == pytest.approx(1.5)
    assert state_data['max_velocity'] == pytest.approx(3.0)
    assert state_data['avg_kinetic_energy'] == pytest.approx(0.75)
    assert state_data['max_kinetic_energy'] == pytest.approx(4.5)
    assert state_data['avg_activity_magnitude'] == pytest.approx(13.0)
    assert state_data['max_activity_magnitude'] == pytest.approx(46.0)
    assert state_data['cells_over_threshold'] == 7
    assert state_data['num_cells'] == expected_num_cells

    assert state_data['control_params']['phoswave_C'] == pytest.approx(expected_C)
    assert state_data['control_params']['electron_D'] == pytest.approx(expected_D)

def test_api_control_success(client, reset_globals):
    """Test: Endpoint /api/control ajusta los parámetros de PhosWave y Electron correctamente."""
    mock_mesh, mock_resonador, mock_electron, mock_agg_state, mock_ctrl_params = reset_globals

    # Usar las constantes mockeadas por reset_globals
    from watchers.watchers_tools.malla_watcher.malla_watcher import BASE_COUPLING_T, BASE_DAMPING_E, K_GAIN_COUPLING, K_GAIN_DAMPING

    # Configurar side_effect para que los mocks actualicen sus atributos C y D
    mock_resonador.ajustar_coeficientes.side_effect = lambda val: setattr(mock_resonador, 'C', max(0.0, val))
    mock_electron.ajustar_coeficientes.side_effect = lambda val: setattr(mock_electron, 'D', max(0.0, val))

    # Test con señal positiva
    signal_pos = 10.0
    expected_C_pos = BASE_COUPLING_T + K_GAIN_COUPLING * signal_pos
    expected_D_pos = BASE_DAMPING_E - K_GAIN_DAMPING * signal_pos

    response_pos = client.post('/api/control', json={'control_signal': signal_pos})
    assert response_pos.status_code == 200
    data_pos = response_pos.get_json()
    assert data_pos['status'] == 'success'
    assert 'Parámetros ajustados' in data_pos['message']

    mock_resonador.ajustar_coeficientes.assert_called_with(pytest.approx(max(0.0, expected_C_pos)))
    mock_electron.ajustar_coeficientes.assert_called_with(pytest.approx(max(0.0, expected_D_pos)))
    assert mock_resonador.C == pytest.approx(max(0.0, expected_C_pos))
    assert mock_electron.D == pytest.approx(max(0.0, expected_D_pos))
    assert mock_ctrl_params['phoswave_C'] == pytest.approx(max(0.0, expected_C_pos))
    assert mock_ctrl_params['electron_D'] == pytest.approx(max(0.0, expected_D_pos))

    # Resetear mocks para la siguiente llamada
    mock_resonador.ajustar_coeficientes.reset_mock()
    mock_electron.ajustar_coeficientes.reset_mock()
    # Restaurar valores base en los mocks para la siguiente prueba de señal
    mock_resonador.C = BASE_COUPLING_T
    mock_electron.D = BASE_DAMPING_E
    mock_ctrl_params.update({"phoswave_C": BASE_COUPLING_T, "electron_D": BASE_DAMPING_E})

    # Test con señal negativa
    signal_neg = -5.0
    expected_C_neg = BASE_COUPLING_T + K_GAIN_COUPLING * signal_neg
    expected_D_neg = BASE_DAMPING_E - K_GAIN_DAMPING * signal_neg

    response_neg = client.post('/api/control', json={'control_signal': signal_neg})
    assert response_neg.status_code == 200
    data_neg = response_neg.get_json()
    assert data_neg['status'] == 'success'

    mock_resonador.ajustar_coeficientes.assert_called_with(pytest.approx(max(0.0, expected_C_neg)))
    mock_electron.ajustar_coeficientes.assert_called_with(pytest.approx(max(0.0, expected_D_neg)))
    assert mock_resonador.C == pytest.approx(max(0.0, expected_C_neg))
    assert mock_electron.D == pytest.approx(max(0.0, expected_D_neg))
    assert mock_ctrl_params['phoswave_C'] == pytest.approx(max(0.0, expected_C_neg))
    assert mock_ctrl_params['electron_D'] == pytest.approx(max(0.0, expected_D_neg))

def test_api_control_invalid_input(client): # No necesita reset_globals
    """Test: Endpoint /api/control maneja correctamente JSON inválido o campos faltantes."""
    response_no_json = client.post('/api/control', data='no es json', content_type='application/json')
    assert response_no_json.status_code == 400
    assert 'Payload JSON vacío, inválido o falta' in response_no_json.get_json()['message']

    response_missing_field = client.post('/api/control', json={'otro_campo': 123})
    assert response_missing_field.status_code == 400
    assert "falta 'control_signal'" in response_missing_field.get_json()['message']

    response_wrong_type = client.post('/api/control', json={'control_signal': 'no_numero'})
    assert response_wrong_type.status_code == 400
    assert "El campo 'control_signal' debe ser un número" in response_wrong_type.get_json()['message']

def test_api_event_pulse(client, reset_globals):
    """Test: Endpoint /api/event aplica correctamente un pulso a la velocidad de una celda."""
    mock_mesh, _, _, _, _ = reset_globals

    cell_coords_q, cell_coords_r = 0, 0
    initial_velocity = 1.5
    pulse_magnitude = 5.0

    # Crear la celda y añadirla al mock de la malla
    target_cell = Cell(cyl_radius=1.0, cyl_theta=1.0, cyl_z=1.0,
                       q_axial=cell_coords_q, r_axial=cell_coords_r,
                       velocity=initial_velocity)
    mock_mesh.cells = {(cell_coords_q, cell_coords_r): target_cell}
    # Configurar el mock de get_cell para que devuelva esta celda
    mock_mesh.get_cell.side_effect = lambda q, r: mock_mesh.cells.get((q, r))

    payload = {'type': 'pulse', 'coords': {'q': cell_coords_q, 'r': cell_coords_r}, 'magnitude': pulse_magnitude}
    response = client.post('/api/event', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert data['message'] == 'Evento procesado'

    # Verificar que la velocidad de la celda se actualizó
    assert target_cell.velocity == pytest.approx(initial_velocity + pulse_magnitude)

def test_api_event_pulse_cell_not_found(client, reset_globals):
    """Test: Endpoint /api/event maneja el caso de celda no encontrada."""
    mock_mesh, _, _, _, _ = reset_globals
    mock_mesh.cells = {} # Asegurar que la malla mockeada está vacía
    mock_mesh.get_cell.return_value = None # get_cell no encontrará nada

    payload = {'type': 'pulse', 'coords': {'q': 99, 'r': 99}, 'magnitude': 1.0}
    response = client.post('/api/event', json=payload)
    assert response.status_code == 503
    data = response.get_json()
    assert data['status'] == 'warning' # O 'error' según la implementación
    assert "Malla no inicializada o vacía" in data['message']

def test_api_event_pulse_cell_not_found_in_populated_mesh(client, reset_globals):
    """
    Test: Endpoint /api/event devuelve 404 si la celda no existe
    en una malla que sí está poblada.
    """
    mock_mesh, _, _, _, _ = reset_globals

    # 1. Configurar la malla mockeada para que NO esté vacía.
    #    Añadir al menos una celda, pero que NO sea la que vamos a buscar.
    existing_cell_coords_q, existing_cell_coords_r = 0, 0
    existing_cell = Cell(cyl_radius=1.0, cyl_theta=0.0, cyl_z=0.0,
                           q_axial=existing_cell_coords_q, r_axial=existing_cell_coords_r)
    mock_mesh.cells = {(existing_cell_coords_q, existing_cell_coords_r): existing_cell}
    # Asegurar que get_all_cells devuelve algo para que la primera verificación en el endpoint pase
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())


    # 2. Configurar el mock de get_cell para que devuelva None cuando se busquen
    #    las coordenadas de la celda no existente.
    #    Y que devuelva la celda existente si se busca esa.
    coords_to_find_q, coords_to_find_r = 99, 99 # Coordenadas que no existen

    def get_cell_side_effect(q, r):
        if (q, r) == (existing_cell_coords_q, existing_cell_coords_r):
            return mock_mesh.cells.get((q,r)) # Devuelve la celda existente
        elif (q, r) == (coords_to_find_q, coords_to_find_r):
            return None # Simula que la celda (99,99) no se encuentra
        return mock_mesh.cells.get((q,r)) # Comportamiento por defecto

    mock_mesh.get_cell.side_effect = get_cell_side_effect

    # 3. Definir el payload para buscar la celda no existente.
    payload = {'type': 'pulse',
               'coords': {'q': coords_to_find_q, 'r': coords_to_find_r},
               'magnitude': 1.0}

    # 4. Realizar la solicitud POST.
    response = client.post('/api/event', json=payload)

    # 5. Verificar la respuesta.
    assert response.status_code == 404 # Esperamos Not Found
    data = response.get_json()
    assert data['status'] == 'warning' # O 'error', según tu implementación para este caso
    assert f"Celda ({coords_to_find_q},{coords_to_find_r}) no encontrada" in data['message']
    # Verificar que la celda existente no fue modificada (opcional, pero bueno)
    assert existing_cell.velocity == 0.0 # Asumiendo que su velocidad inicial era 0.0 y no se aplicó pulso

def test_api_malla(client, reset_globals):
    """Test: Endpoint /api/malla devuelve la estructura completa de la malla y sus celdas."""
    mock_mesh, _, _, _, _ = reset_globals

    # Configurar celdas específicas y metadatos en el mock_mesh para este test
    cell_a = Cell(cyl_radius=mock_mesh.radius, cyl_theta=0.1, cyl_z=0.5, q_axial=0, r_axial=0, amplitude=1, velocity=0.1, q_vector=np.array([0.1,0.1]))
    cell_b = Cell(cyl_radius=mock_mesh.radius, cyl_theta=0.2, cyl_z=1.5, q_axial=1, r_axial=0, amplitude=2, velocity=0.2, q_vector=np.array([0.2,0.2]))
    mock_mesh.cells = {(0,0): cell_a, (1,0): cell_b}
    mock_mesh.get_all_cells.return_value = list(mock_mesh.cells.values())
    # Actualizar min_z y max_z del mock basado en las celdas añadidas
    mock_mesh.min_z = 0.5
    mock_mesh.max_z = 1.5

    response = client.get('/api/malla')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'

    metadata = data['metadata']
    assert metadata['radius'] == mock_mesh.radius
    assert metadata['num_cells'] == 2
    assert metadata['periodic_z'] == mock_mesh.periodic_z
    assert metadata['z_bounds'] == {"min": 0.5, "max": 1.5}

    cells_data = data['cells']
    assert len(cells_data) == 2
    # Verificar que los datos de las celdas corresponden a cell_a y cell_b
    # (puedes hacer aserciones más detalladas si es necesario)
    assert any(cd['axial_coords']['q'] == 0 and cd['amplitude'] == 1 for cd in cells_data)
    assert any(cd['axial_coords']['q'] == 1 and cd['amplitude'] == 2 for cd in cells_data)
    for cell_data_item in cells_data:
        assert 'q_vector' in cell_data_item
        assert isinstance(cell_data_item['q_vector'], list)

def test_api_config(client, reset_globals):
    """Test: Endpoint /api/config devuelve la configuración actual del módulo, incluyendo valores de entorno mockeados."""
    mock_mesh, mock_resonador, mock_electron, mock_agg_state, mock_ctrl_params = reset_globals

    # 1. Definir los valores ESPERADOS que el endpoint /api/config debería devolver
    #    para la sección 'malla_config' (la que lee de os.environ).
    #    Estos deben coincidir con lo que mockeamos para las constantes del módulo en reset_globals
    #    si queremos que las aserciones directas con las constantes del módulo funcionen.
    expected_malla_config_values = {
        "MW_RADIUS": "7.7",  # Valor que MESH_RADIUS fue mockeado en reset_globals
        "MW_HEIGHT_SEG": "2",   # Valor que MESH_HEIGHT_SEGMENTS fue mockeado
        "MW_CIRCUM_SEG": "11",  # Valor que MESH_CIRCUMFERENCE_SEGMENTS fue mockeado
        "MW_HEX_SIZE": "0.8", # Valor que MESH_HEX_SIZE fue mockeado
        "MW_PERIODIC_Z": "False" # Valor que MESH_PERIODIC_Z fue mockeado
    }

    # 2. Crear un side_effect para os.environ.get
    #    Guardar el os.environ.get original para llamarlo para claves no mockeadas (buena práctica)
    original_os_environ_get = os.environ.get 
    def mock_environ_get_for_config(key, default=None):
        if key in expected_malla_config_values:
            return expected_malla_config_values[key]
        return original_os_environ_get(key, default) # Llama al original para otras claves

    # 3. Aplicar el patch a os.environ.get DENTRO de este test
    with patch('os.environ.get', side_effect=mock_environ_get_for_config):
        # Importar las constantes DEL MÓDULO (que fueron mockeadas por reset_globals)
        # Estas se usarán para verificar las secciones del config que SÍ usan constantes del módulo.
        from watchers.watchers_tools.malla_watcher.malla_watcher import (
            # MESH_RADIUS, MESH_HEIGHT_SEGMENTS, etc. NO se usan para malla_config aquí,
            # porque compararemos con expected_malla_config_values.
            # Pero sí para otras secciones:
            MATRIZ_ECU_BASE_URL, TORUS_NUM_CAPAS, TORUS_NUM_FILAS, TORUS_NUM_COLUMNAS,
            AMPLITUDE_INFLUENCE_THRESHOLD, MAX_AMPLITUDE_FOR_NORMALIZATION, SIMULATION_INTERVAL,
            BASE_COUPLING_T, BASE_DAMPING_E, K_GAIN_COUPLING, K_GAIN_DAMPING, DPHI_DT_INFLUENCE_THRESHOLD
        )

        # Configurar valores actuales de C y D en los mocks para que el endpoint los lea
        mock_resonador.C = 0.88
        mock_electron.D = 0.12

        response = client.get('/api/config')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        config = data['config']

        # Verificar malla_config (ahora comparamos con los valores esperados que os.environ.get devolvió)
        m_cfg = config['malla_config']
        assert m_cfg['radius'] == float(expected_malla_config_values["MW_RADIUS"])
        assert m_cfg['height_segments'] == int(expected_malla_config_values["MW_HEIGHT_SEG"])
        assert m_cfg['circumference_segments_target'] == int(expected_malla_config_values["MW_CIRCUM_SEG"])
        assert m_cfg['circumference_segments_actual'] == mock_mesh.circumference_segments_actual # Esto viene del mock_mesh
        assert m_cfg['hex_size'] == float(expected_malla_config_values["MW_HEX_SIZE"])
        assert m_cfg['periodic_z'] == (expected_malla_config_values["MW_PERIODIC_Z"].lower() == "true")


        # Verificar communication_config (estas usan constantes del módulo mockeadas por reset_globals)
        comm_cfg = config['communication_config']
        assert comm_cfg['matriz_ecu_url'] == MATRIZ_ECU_BASE_URL
        assert comm_cfg['torus_dims'] == f"{TORUS_NUM_CAPAS}x{TORUS_NUM_FILAS}x{TORUS_NUM_COLUMNAS}"
        assert comm_cfg['influence_threshold'] == AMPLITUDE_INFLUENCE_THRESHOLD
        assert comm_cfg['max_activity_normalization'] == MAX_AMPLITUDE_FOR_NORMALIZATION

        # Verificar simulation_config
        sim_cfg = config['simulation_config']
        assert sim_cfg['interval'] == SIMULATION_INTERVAL
        assert sim_cfg['dphi_dt_influence_threshold'] == DPHI_DT_INFLUENCE_THRESHOLD

        # Verificar control_config
        ctrl_cfg = config['control_config']
        assert ctrl_cfg['base_coupling_t'] == BASE_COUPLING_T
        assert ctrl_cfg['base_damping_e'] == BASE_DAMPING_E
        assert ctrl_cfg['k_gain_coupling'] == K_GAIN_COUPLING
        assert ctrl_cfg['k_gain_damping'] == K_GAIN_DAMPING
        assert ctrl_cfg['current_coupling_C'] == pytest.approx(0.88)
        assert ctrl_cfg['current_damping_D'] == pytest.approx(0.12)

# NUEVO TEST: test_api_malla_influence_push
def test_api_malla_influence_push(client, reset_globals):
    """Test: Endpoint /api/malla/influence (push) aplica correctamente un campo vectorial externo."""
    mock_mesh, _, _, _, _ = reset_globals # Solo necesitamos mock_mesh

    # Asegurar que la malla mockeada tiene celdas para que el endpoint no falle por malla vacía
    if not mock_mesh.cells:
        mock_mesh.cells[(0,0)] = Cell(cyl_radius=1.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0, r_axial=0)

    test_field_vector_payload = [
        [[[1.1, 2.2]], [[3.3, 4.4]]], # Capa 0, 2 filas, 1 columna, vector 2D
        [[[5.5, 6.6]], [[7.7, 8.8]]]  # Capa 1, 2 filas, 1 columna, vector 2D
    ]

    # Mockear la función de módulo apply_external_field_to_mesh
    with patch('watchers.watchers_tools.malla_watcher.malla_watcher.apply_external_field_to_mesh') as mock_apply_func:
        response = client.post('/api/malla/influence', json={'field_vector': test_field_vector_payload})

        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'success'
        assert 'Campo vectorial externo (push) aplicado' in data['message']

        # Verificar que la función mockeada fue llamada con la instancia de malla global y el payload
        mock_apply_func.assert_called_once_with(mock_mesh, test_field_vector_payload)

def test_api_malla_influence_push_invalid_payload(client, reset_globals):
    """Test: Endpoint /api/malla/influence (push) maneja payloads inválidos."""
    mock_mesh, _, _, _, _ = reset_globals
    if not mock_mesh.cells: # Asegurar que la malla no está vacía para que el error sea por payload
        mock_mesh.cells[(0,0)] = Cell(cyl_radius=1.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0, r_axial=0)

    response_missing_key = client.post('/api/malla/influence', json={'otro_dato': 'valor'})
    assert response_missing_key.status_code == 400
    data_missing = response_missing_key.get_json()
    assert data_missing['status'] == 'error'
    assert "falta 'field_vector'" in data_missing['message'].lower()

    response_empty_json = client.post('/api/malla/influence', json={})
    assert response_empty_json.status_code == 400
    data_empty = response_empty_json.get_json()
    assert data_empty['status'] == 'error'
    assert "falta 'field_vector'" in data_empty['message'].lower() # O el mensaje de payload vacío

# --- END OF FILE test_malla_watcher.py (REFINADO para Osciladores Acoplados) ---