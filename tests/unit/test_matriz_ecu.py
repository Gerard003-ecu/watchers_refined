# --- START OF FILE test_matriz_ecu.py ---

# tests/unit/test_matriz_ecu.py

import logging
import threading

import numpy as np
import pytest
from flask.testing import FlaskClient

from ecu.matriz_ecu import (
    NUM_CAPAS,
    NUM_COLUMNAS,
    NUM_FILAS,
    ToroidalField,
    app,
    campo_toroidal_global_servicio,
)

logger = logging.getLogger(__name__)


# --- Fixtures ---
@pytest.fixture
def campo_toroidal_test():
    """Fixture: Campo toroidal aislado para tests unitarios."""
    return ToroidalField(
        num_capas=2,
        num_rows=3,
        num_cols=4,
        propagation_coeffs=None,
        dissipation_coeffs=None,
    )


@pytest.fixture
def cliente_flask() -> FlaskClient:
    """Fixture: Cliente de pruebas de Flask."""
    app.config["TESTING"] = True
    with app.test_client() as cliente:
        with campo_toroidal_global_servicio.lock:
            campo_toroidal_global_servicio.campo_q = [
                np.zeros((NUM_FILAS, NUM_COLUMNAS), dtype=np.complex128)
                for _ in range(NUM_CAPAS)
            ]
        logger.debug("Estado del campo global reseteado para test API")
        yield cliente


# --- Tests para ToroidalField (Usando instancia aislada 'campo_toroidal_test')
def test_inicializacion_valida(campo_toroidal_test: ToroidalField):
    """Test: Creación OK, params válidos (instancia aislada)."""
    tf = campo_toroidal_test
    assert tf.num_capas == 2
    assert tf.num_rows == 3
    assert tf.num_cols == 4
    assert isinstance(tf.campo_q, list)
    assert len(tf.campo_q) == 2
    assert tf.campo_q[0].shape == (3, 4)
    assert np.all(tf.campo_q[0] == 0)
    assert len(tf.propagation_coeffs) == 2
    assert len(tf.dissipation_coeffs) == 2
    assert isinstance(tf.lock, type(threading.Lock()))


def test_inicializacion_con_params_capa():
    """Test: Creación con propagation_coeffs y dissipation_coeffs específicos."""
    propagation_coeffs_test = [0.1, 0.2]
    dissipation_coeffs_test = [0.01, 0.02]
    tf = ToroidalField(
        num_capas=2,
        num_rows=2,
        num_cols=2,
        propagation_coeffs=propagation_coeffs_test,
        dissipation_coeffs=dissipation_coeffs_test,
    )
    assert tf.propagation_coeffs == propagation_coeffs_test
    assert tf.dissipation_coeffs == dissipation_coeffs_test


def test_inicializacion_params_capa_longitud_incorrecta():
    """Test: Error al crear con listas de params de longitud incorrecta."""
    with pytest.raises(
        ValueError, match="La lista 'propagation_coeffs' debe tener longitud 2"
    ):
        ToroidalField(num_capas=2, num_rows=2, num_cols=2, propagation_coeffs=[0.1])

    with pytest.raises(
        ValueError, match="La lista 'dissipation_coeffs' debe tener longitud 3"
    ):
        ToroidalField(
            num_capas=3, num_rows=2, num_cols=2, dissipation_coeffs=[0.1, 0.2]
        )


def test_inicializacion_invalida():
    """Test: Error al crear con dimensiones inválidas."""
    with pytest.raises(ValueError, match="dimensiones .* deben ser positivas"):
        ToroidalField(num_capas=0, num_rows=2, num_cols=2)

    with pytest.raises(ValueError, match="dimensiones .* deben ser positivas"):
        ToroidalField(num_capas=1, num_rows=-1, num_cols=2)

    with pytest.raises(ValueError, match="dimensiones .* deben ser positivas"):
        ToroidalField(num_capas=1, num_rows=2, num_cols=0)


def test_aplicar_influencia_valida(campo_toroidal_test: ToroidalField):
    """Test: Aplicar influencia en posición válida."""
    tf = campo_toroidal_test
    capa, row, col = 0, 1, 2
    vector = 0.5 - 0.3j
    with tf.lock:
        valor_inicial = np.copy(tf.campo_q[capa][row, col])

    success = tf.aplicar_influencia(capa, row, col, vector, "test_watcher")
    assert success is True

    with tf.lock:
        valor_final = tf.campo_q[capa][row, col]
    assert np.isclose(valor_final, valor_inicial + vector)


def test_aplicar_influencia_fuera_rango(campo_toroidal_test: ToroidalField, caplog):
    """Test: Manejo de índices fuera de rango."""
    tf = campo_toroidal_test
    vector = 1.0 + 0.0j
    with tf.lock:
        valor_original = [np.copy(c) for c in tf.campo_q]

    with caplog.at_level(logging.ERROR):
        success = tf.aplicar_influencia(5, 1, 1, vector, "watcher_err_capa")
    assert success is False
    assert "índice de capa fuera de rango (5)" in caplog.text.lower()
    with tf.lock:
        for i in range(tf.num_capas):
            assert np.array_equal(tf.campo_q[i], valor_original[i])
    caplog.clear()

    with caplog.at_level(logging.ERROR):
        success = tf.aplicar_influencia(0, 5, 2, vector, "watcher_err_fila")
    assert success is False
    assert "índice de fila fuera de rango (5)" in caplog.text.lower()
    with tf.lock:
        for i in range(tf.num_capas):
            assert np.array_equal(tf.campo_q[i], valor_original[i])
    caplog.clear()

    with caplog.at_level(logging.ERROR):
        success = tf.aplicar_influencia(0, 1, -1, vector, "watcher_err_col")
    assert success is False
    assert "índice de columna fuera de rango (-1)" in caplog.text.lower()
    with tf.lock:
        for i in range(tf.num_capas):
            assert np.array_equal(tf.campo_q[i], valor_original[i])


def test_aplicar_influencia_vector_invalido(campo_toroidal_test: ToroidalField, caplog):
    """Test: Aplicar influencia con vector de formato incorrecto."""
    tf = campo_toroidal_test
    with tf.lock:
        valor_original = [np.copy(c) for c in tf.campo_q]

    with caplog.at_level(logging.ERROR):
        success = tf.aplicar_influencia(
            0, 1, 1, "not a complex number", "watcher_vec_err"
        )
    assert success is False
    assert "vector de influencia inválido" in caplog.text.lower()
    assert "debe ser un número complejo" in caplog.text.lower()
    with tf.lock:
        for i in range(tf.num_capas):
            assert np.array_equal(tf.campo_q[i], valor_original[i])




def test_get_energy_density_map(campo_toroidal_test: ToroidalField):
    """Test: Mapa de densidad de energía con pesos por capa."""
    tf = campo_toroidal_test
    tf.aplicar_influencia(0, 1, 1, 3.0 + 4.0j, "test_uni1_capa0")
    tf.aplicar_influencia(1, 1, 1, 0.0 - 6.0j, "test_uni2_capa1")

    energy_map = tf.get_energy_density_map()
    assert energy_map.shape == (3, 4)
    # Los pesos por defecto para 2 capas son 1.0 y 0.5
    expected_value_11 = (1.0 * np.abs(3.0 + 4.0j) ** 2) + (
        0.5 * np.abs(0.0 - 6.0j) ** 2
    )
    assert energy_map[1, 1] == pytest.approx(expected_value_11)
    assert energy_map[0, 0] == pytest.approx(0.0)


def test_get_energy_density_map_una_capa():
    """Test: Mapa de densidad de energía con una sola capa."""
    tf = ToroidalField(num_capas=1, num_rows=2, num_cols=2)
    tf.aplicar_influencia(0, 0, 0, 1.0 + 1.0j, "test_uni_1capa")
    energy_map = tf.get_energy_density_map()
    assert energy_map.shape == (2, 2)
    assert energy_map[0, 0] == pytest.approx(np.abs(1.0 + 1.0j) ** 2)
    assert energy_map[0, 1] == pytest.approx(0.0)


def test_apply_wave_dynamics_step(campo_toroidal_test: ToroidalField):
    """Test: Paso de dinámica de ondas (advección, acoplamiento, disipación)."""
    tf = campo_toroidal_test
    initial_vector = 1.0 + 2.0j
    capa, row, col = 0, 1, 1
    tf.aplicar_influencia(capa, row, col, initial_vector, "test_rot_watcher")

    dt, beta = 0.1, 0.1

    with tf.lock:
        initial_state_full = [np.copy(c) for c in tf.campo_q]

    tf.apply_wave_dynamics_step(dt, beta)

    with tf.lock:
        final_state_full = tf.campo_q

    assert not np.allclose(
        final_state_full[capa][row, col], initial_state_full[capa][row, col]
    )
    assert np.all(np.isfinite(final_state_full[capa][row, col]))


# --- Tests para la API REST (Usando instancia global y cliente_flask) ---
def test_endpoint_health(cliente_flask: FlaskClient):
    """Test: Endpoint /api/health (sim inactiva)."""
    respuesta = cliente_flask.get("/api/health")
    datos = respuesta.get_json()
    assert respuesta.status_code == 503
    assert datos["status"] == "warning"
    assert datos["field_initialized"] is True
    assert datos["simulation_running"] is False


def test_endpoint_energy_api(cliente_flask: FlaskClient):
    """Test: Respuesta OK del endpoint /api/ecu/energy."""
    influence_payload = {
        "capa": 0,
        "row": 1,
        "col": 2,
        "vector": [3.0, 4.0],
        "nombre_watcher": "test_api_energy",
    }
    vector_influencia = 3.0 + 4.0j
    cliente_flask.post("/api/ecu/influence", json=influence_payload)

    get_response = cliente_flask.get("/api/ecu/energy")
    assert get_response.status_code == 200
    data = get_response.get_json()

    assert data["status"] == "success"
    assert "data" in data
    assert "metadata" in data
    assert data["data"]["type"] == "energy_map"
    assert "energy_density_map" in data["data"]
    assert data["metadata"]["layers"] == NUM_CAPAS

    # Verificar que el valor de energía es correcto
    energy_map = data["data"]["energy_density_map"]
    # El peso para la capa 0 es 1.0 por defecto
    expected_energy = np.abs(vector_influencia) ** 2
    assert energy_map[1][2] == pytest.approx(expected_energy)


def test_endpoint_influence_valido(cliente_flask: FlaskClient):
    """Test: Aplicar influencia válida vía API."""
    payload = {
        "capa": 1,
        "row": 2,
        "col": 3,
        "vector": [5.0, -1.0],
        "nombre_watcher": "api_test_influence",
    }
    capa_idx = payload["capa"]
    row_idx = payload["row"]
    col_idx = payload["col"]
    vector = 5.0 - 1.0j

    with campo_toroidal_global_servicio.lock:
        valor_inicial = np.copy(
            campo_toroidal_global_servicio.campo_q[capa_idx][row_idx, col_idx]
        )

    respuesta = cliente_flask.post("/api/ecu/influence", json=payload)
    assert respuesta.status_code == 200
    datos = respuesta.get_json()
    assert datos["status"] == "success"
    assert datos["applied_to"]["capa"] == capa_idx
    assert datos["applied_to"]["row"] == row_idx
    assert datos["applied_to"]["col"] == col_idx
    assert datos["vector"] == payload["vector"]

    with campo_toroidal_global_servicio.lock:
        valor_final = campo_toroidal_global_servicio.campo_q[capa_idx][row_idx, col_idx]
    assert np.isclose(valor_final, valor_inicial + vector)


def test_endpoint_influence_invalido_datos(cliente_flask: FlaskClient):
    """Test: Errores 400 por datos inválidos en API influence."""
    respuesta = cliente_flask.post("/api/ecu/influence", json={})
    assert respuesta.status_code == 400
    assert "payload json vacío o ausente" in respuesta.get_json()["message"].lower()

    payload_faltan = {"capa": 0, "row": 1}
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_faltan)
    assert respuesta.status_code == 400
    # fmt: off
    assert "faltan campos" in respuesta.get_json()["message"].lower()
    # fmt: on

    payload_tipo_err = {
        "capa": "cero",
        "row": 1,
        "col": 1,
        "vector": [1, 1],
        "nombre_watcher": "t",
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_tipo_err)
    assert respuesta.status_code == 400
    assert "error en el formato de los datos" in respuesta.get_json()["message"].lower()

    payload_vec_err_len = {
        "capa": 0,
        "row": 1,
        "col": 1,
        "vector": [1],
        "nombre_watcher": "t",
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_vec_err_len)
    assert respuesta.status_code == 400
    assert "formato de vector inválido" in respuesta.get_json()["message"].lower()

    payload_vec_err_type = {
        "capa": 0,
        "row": 1,
        "col": 1,
        "vector": [1, "a"],
        "nombre_watcher": "t",
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_vec_err_type)
    assert respuesta.status_code == 400
    assert "números válidos" in respuesta.get_json()["message"].lower()


def test_endpoint_influence_invalido_rango(cliente_flask: FlaskClient):
    """Test: Error 400 por API índic. fuera de rango."""
    payload_capa = {
        "capa": NUM_CAPAS,
        "row": 0,
        "col": 0,
        "vector": [1.0, 0.0],
        "nombre_watcher": "api_test_range_capa",
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_capa)
    assert respuesta.status_code == 400
    assert "capa" in respuesta.get_json()["message"].lower()
    assert "fuera de rango" in respuesta.get_json()["message"].lower()

    payload_row = {
        "capa": 0,
        "row": NUM_FILAS,
        "col": 0,
        "vector": [1.0, 0.0],
        "nombre_watcher": "api_test_range_row",
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_row)
    assert respuesta.status_code == 400
    assert "fila" in respuesta.get_json()["message"].lower()
    assert "fuera de rango" in respuesta.get_json()["message"].lower()

    payload_col = {
        "capa": 0,
        "row": 0,
        "col": NUM_COLUMNAS,
        "vector": [1.0, 0.0],
        "nombre_watcher": "api_test_range_col",
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_col)
    assert respuesta.status_code == 400
    assert "columna" in respuesta.get_json()["message"].lower()
    assert "fuera de rango" in respuesta.get_json()["message"].lower()


def test_endpoint_get_field_vector_paginated(cliente_flask: FlaskClient):
    """Test: Paginación en endpoint /api/ecu/field_vector."""
    # Probar paginación de capas
    response = cliente_flask.get("/api/ecu/field_vector?page=1&per_page=2")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["data"]) == 2  # 2 capas por página
    assert data["pagination"]["page"] == 1
    assert data["pagination"]["per_page"] == 2
    assert data["pagination"]["total_items"] == NUM_CAPAS
    assert data["pagination"]["total_pages"] == (NUM_CAPAS + 1) // 2

    # Probar paginación de filas dentro de una capa
    response = cliente_flask.get("/api/ecu/field_vector?layer=0&page=2&per_page=1")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert len(data["data"]) == 1  # 1 fila por página
    assert len(data["data"][0]) == NUM_COLUMNAS
    assert data["pagination"]["page"] == 2
    assert data["pagination"]["total_items"] == NUM_FILAS


def test_endpoint_get_field_vector_errors(cliente_flask: FlaskClient):
    """Test: Errores en endpoint /api/ecu/field_vector."""
    # Capa fuera de rango
    response = cliente_flask.get("/api/ecu/field_vector?layer=99")
    assert response.status_code == 400
    assert "fuera de rango" in response.get_json()["message"]

    # Página fuera de rango
    response = cliente_flask.get("/api/ecu/field_vector?page=99")
    assert response.status_code == 404
    assert "Página fuera de rango" in response.get_json()["message"]


def test_endpoint_energy_error_interno(cliente_flask: FlaskClient, mocker):
    """Test: Manejo de errores internos en API /api/ecu/energy (500)."""
    mocker.patch(
        "ecu.matriz_ecu.campo_toroidal_global_servicio.get_energy_density_map",
        side_effect=Exception("Mock error interno"),
    )
    respuesta = cliente_flask.get("/api/ecu/energy")
    assert respuesta.status_code == 500
    datos = respuesta.get_json()
    assert datos["status"] == "error"
    assert "error interno" in datos["message"].lower()


def test_set_uniform_potential_field(campo_toroidal_test: ToroidalField):
    """Test: Inicialización del campo a potencial uniforme con fase aleatoria."""
    tf = campo_toroidal_test
    seed = 42
    tf.set_uniform_potential_field(seed)

    # Verificar que todos los valores son números complejos con magnitud ~1
    for capa in tf.campo_q:
        magnitudes = np.abs(capa)
        assert np.allclose(magnitudes, 1.0)

    # Verificar que con la misma semilla, el resultado es idéntico
    tf2 = ToroidalField(
        tf.num_capas,
        tf.num_rows,
        tf.num_cols,
        propagation_coeffs=tf.propagation_coeffs,
        dissipation_coeffs=tf.dissipation_coeffs,
    )
    tf2.set_uniform_potential_field(seed)
    for capa1, capa2 in zip(tf.campo_q, tf2.campo_q, strict=False):
        assert np.allclose(capa1, capa2)


def test_apply_internal_phase_evolution(campo_toroidal_test: ToroidalField):
    """Test: Aplicación de un paso de evolución de fase interna."""
    tf = campo_toroidal_test
    tf.set_uniform_potential_field(seed=123)
    initial_phases = [np.copy(capa) for capa in tf.campo_q]

    dt = 0.1
    tf.apply_internal_phase_evolution(dt)

    for i, initial_capa in enumerate(initial_phases):
        prop_coeff = tf.propagation_coeffs[i]
        expected_phase_change = np.exp(-1j * prop_coeff * dt)
        expected_final_capa = initial_capa * expected_phase_change
        assert np.allclose(tf.campo_q[i], expected_final_capa)
