# --- START OF FILE test_matriz_ecu.py ---

# tests/unit/test_matriz_ecu.py

import pytest
import numpy as np
import threading
from ecu.matriz_ecu import (
    ToroidalField,
    app,
    campo_toroidal_global_servicio,
    NUM_CAPAS,
    NUM_FILAS,
    NUM_COLUMNAS,
)
from flask.testing import FlaskClient
import logging

logger = logging.getLogger(__name__)


# --- Fixtures ---
@pytest.fixture
def campo_toroidal_test():
    """Fixture: Campo toroidal aislado para tests unitarios."""
    return ToroidalField(
        num_capas=2, num_rows=3, num_cols=4, alphas=None, dampings=None
    )


@pytest.fixture
def cliente_flask() -> FlaskClient:
    """Fixture: Cliente de pruebas de Flask."""
    app.config['TESTING'] = True
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
    assert len(tf.alphas) == 2
    assert len(tf.dampings) == 2
    assert isinstance(tf.lock, type(threading.Lock()))


def test_inicializacion_con_params_capa():
    """Test: Creación con alphas y dampings específicos."""
    alphas_test = [0.1, 0.2]
    dampings_test = [0.01, 0.02]
    tf = ToroidalField(
        num_capas=2,
        num_rows=2,
        num_cols=2,
        alphas=alphas_test,
        dampings=dampings_test
    )
    assert tf.alphas == alphas_test
    assert tf.dampings == dampings_test


def test_inicializacion_params_capa_longitud_incorrecta():
    """Test: Error al crear con listas de params de longitud incorrecta."""
    with pytest.raises(
        ValueError, match="La lista 'alphas' debe tener longitud 2"
    ):
        ToroidalField(num_capas=2, num_rows=2, num_cols=2, alphas=[0.1])

    with pytest.raises(
        ValueError, match="La lista 'dampings' debe tener longitud 3"
    ):
        ToroidalField(num_capas=3, num_rows=2, num_cols=2, dampings=[0.1, 0.2])


def test_inicializacion_invalida():
    """Test: Error al crear con dimensiones inválidas."""
    with pytest.raises(
        ValueError, match="dimensiones .* deben ser positivas"
    ):
        ToroidalField(num_capas=0, num_rows=2, num_cols=2)

    with pytest.raises(
        ValueError, match="dimensiones .* deben ser positivas"
    ):
        ToroidalField(num_capas=1, num_rows=-1, num_cols=2)

    with pytest.raises(
        ValueError, match="dimensiones .* deben ser positivas"
    ):
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


def test_aplicar_influencia_fuera_rango(
    campo_toroidal_test: ToroidalField, caplog
):
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


def test_aplicar_influencia_vector_invalido(
    campo_toroidal_test: ToroidalField, caplog
):
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


def test_get_neighbors_conectividad_toroidal(
    campo_toroidal_test: ToroidalField
):
    """Test: Vecinos con wraparound toroidal."""
    tf = campo_toroidal_test
    vecinos_00 = tf.get_neighbors(0, 0)
    expected_00 = [(2, 0), (1, 0), (0, 3), (0, 1)]
    assert set(vecinos_00) == set(expected_00), "Error en vecinos de (0,0)"

    vecinos_23 = tf.get_neighbors(2, 3)
    expected_23 = [(1, 3), (0, 3), (2, 2), (2, 0)]
    assert set(vecinos_23) == set(expected_23), "Error en vecinos de (2,3)"


def test_calcular_gradiente_adaptativo(
    campo_toroidal_test: ToroidalField
):
    """Test: Cálculo de gradiente entre capas."""
    tf = campo_toroidal_test
    tf.aplicar_influencia(0, 1, 1, 3.0 + 4.0j, "test_grad1_capa0")
    tf.aplicar_influencia(1, 1, 1, 0.0 - 6.0j, "test_grad1_capa1")

    gradiente = tf.calcular_gradiente_adaptativo()
    assert gradiente.shape == (1, 3, 4)

    expected_diff_mag = np.abs((3.0 + 4.0j) - (0.0 - 6.0j))
    assert gradiente[0, 1, 1] == pytest.approx(expected_diff_mag)
    assert gradiente[0, 0, 0] == pytest.approx(0.0)


def test_calcular_gradiente_sin_suficientes_capas():
    """Test: Gradiente con menos de 2 capas."""
    tf = ToroidalField(num_capas=1, num_rows=2, num_cols=2)
    gradiente = tf.calcular_gradiente_adaptativo()
    assert gradiente.size == 0
    assert gradiente.shape == (0, )


def test_obtener_campo_unificado(campo_toroidal_test: ToroidalField):
    """Test: Campo unificado con pesos por capa."""
    tf = campo_toroidal_test
    tf.aplicar_influencia(0, 1, 1, 3.0 + 4.0j, "test_uni1_capa0")
    tf.aplicar_influencia(1, 1, 1, 0.0 - 6.0j, "test_uni2_capa1")

    campo_uni = tf.obtener_campo_unificado()
    assert campo_uni.shape == (3, 4)
    expected_value_11 = 1.0 * np.abs(3.0 + 4.0j) + 0.5 * np.abs(0.0 - 6.0j)
    assert campo_uni[1, 1] == pytest.approx(expected_value_11)
    assert campo_uni[0, 0] == pytest.approx(0.0)


def test_obtener_campo_unificado_una_capa():
    """Test: Campo unificado con una sola capa."""
    tf = ToroidalField(num_capas=1, num_rows=2, num_cols=2)
    tf.aplicar_influencia(0, 0, 0, 1.0 + 1.0j, "test_uni_1capa")
    campo_uni = tf.obtener_campo_unificado()
    assert campo_uni.shape == (2, 2)
    assert campo_uni[0, 0] == pytest.approx(np.abs(1.0 + 1.0j))
    assert campo_uni[0, 1] == pytest.approx(0.0)


def test_apply_rotational_step(campo_toroidal_test: ToroidalField):
    """Test: Paso rotacional (advección, acoplamiento, disipación)."""
    tf = campo_toroidal_test
    initial_vector = 1.0 + 2.0j
    capa, row, col = 0, 1, 1
    tf.aplicar_influencia(capa, row, col, initial_vector, "test_rot_watcher")

    dt, beta = 0.1, 0.1

    with tf.lock:
        initial_state_full = [np.copy(c) for c in tf.campo_q]

    tf.apply_rotational_step(dt, beta)

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


def test_endpoint_ecu_api(cliente_flask: FlaskClient):
    """Test: Respuesta OK del endpoint /api/ecu."""
    # Paso 1: Definir y aplicar influencias en múltiples capas
    influence_payload_1 = {
        "capa": 0, "row": 0, "col": 0, "vector": [3.0, 4.0],
        "nombre_watcher": "test_api_ecu_1"
    }
    influence_payload_2 = {
        "capa": 1, "row": 0, "col": 0, "vector": [0.0, -6.0],
        "nombre_watcher": "test_api_ecu_2"
    }
    vector_influencia_1 = 3.0 + 4.0j
    vector_influencia_2 = 0.0 - 6.0j

    # Aplicar influencias
    cliente_flask.post("/api/ecu/influence", json=influence_payload_1)
    cliente_flask.post("/api/ecu/influence", json=influence_payload_2)

    # Paso 2: Usar cliente_flask.get para obtener el estado del ECU
    get_response = cliente_flask.get("/api/ecu")
    assert get_response.status_code == 200
    get_data = get_response.get_json()
    assert get_data["status"] == "success"
    assert "estado_campo_unificado" in get_data
    assert get_data["metadata"]["capas"] == NUM_CAPAS

    # Paso 3: Calcular el valor unificado esperado
    pesos = (
        np.linspace(1.0, 0.5, NUM_CAPAS)
        if NUM_CAPAS > 1
        else np.array([1.0])
    )
    valor_esperado = (pesos[0] * np.abs(vector_influencia_1)) + \
                     (pesos[1] * np.abs(vector_influencia_2))

    # Paso 4: Realizar la aserción
    assert get_data["estado_campo_unificado"][0][0] == pytest.approx(valor_esperado)


def test_endpoint_influence_valido(cliente_flask: FlaskClient):
    """Test: Aplicar influencia válida vía API."""
    payload = {
        "capa": 1,
        "row": 2,
        "col": 3,
        "vector": [5.0, -1.0],
        "nombre_watcher": "api_test_influence"
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
        valor_final = campo_toroidal_global_servicio.campo_q[capa_idx][
            row_idx, col_idx
        ]
    assert np.isclose(valor_final, valor_inicial + vector)


def test_endpoint_influence_invalido_datos(cliente_flask: FlaskClient):
    """Test: Errores 400 por datos inválidos en API influence."""
    respuesta = cliente_flask.post("/api/ecu/influence", json={})
    assert respuesta.status_code == 400
    assert "payload json vacío o ausente" in respuesta.get_json()[
        "message"].lower()

    payload_faltan = {"capa": 0, "row": 1}
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_faltan)
    assert respuesta.status_code == 400
    # fmt: off
    assert "faltan campos" in respuesta.get_json()["message"].lower()
    # fmt: on

    payload_tipo_err = {
        "capa": "cero", "row": 1, "col": 1, "vector": [1, 1],
        "nombre_watcher": "t"}
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_tipo_err)
    assert respuesta.status_code == 400
    assert "errores de tipo" in respuesta.get_json()["message"].lower()

    payload_vec_err_len = {
        "capa": 0, "row": 1, "col": 1, "vector": [1], "nombre_watcher": "t"
    }
    respuesta = cliente_flask.post(
        "/api/ecu/influence", json=payload_vec_err_len
    )
    assert respuesta.status_code == 400
    assert "campo 'vector' debe ser una lista de 2 elementos" in respuesta.get_json()["message"].lower()

    payload_vec_err_type = {
        "capa": 0, "row": 1, "col": 1, "vector": [
            1, "a"], "nombre_watcher": "t"}
    respuesta = cliente_flask.post(
        "/api/ecu/influence", json=payload_vec_err_type
    )
    assert respuesta.status_code == 400
    assert "elementos del 'vector' deben ser números" in respuesta.get_json()["message"].lower()


def test_endpoint_influence_invalido_rango(cliente_flask: FlaskClient):
    """Test: Error 400 por API índic. fuera de rango."""
    payload_capa = {
        "capa": NUM_CAPAS, "row": 0, "col": 0, "vector": [1.0, 0.0],
        "nombre_watcher": "api_test_range_capa"
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_capa)
    assert respuesta.status_code == 400
    assert "índice de capa fuera de rango" in respuesta.get_json()["message"].lower()

    payload_row = {
        "capa": 0, "row": NUM_FILAS, "col": 0, "vector": [1.0, 0.0],
        "nombre_watcher": "api_test_range_row"
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_row)
    assert respuesta.status_code == 400
    assert "índice de fila fuera de rango" in respuesta.get_json()["message"].lower()

    payload_col = {
        "capa": 0, "row": 0, "col": NUM_COLUMNAS, "vector": [1.0, 0.0],
        "nombre_watcher": "api_test_range_col"
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_col)
    assert respuesta.status_code == 400
    assert "índice de columna fuera de rango" in respuesta.get_json()["message"].lower()


def test_endpoint_get_field_vector(cliente_flask):
    """Test: Endpoint /api/ecu/field_vector (GET)."""
    from ecu.matriz_ecu import (
        campo_toroidal_global_servicio, NUM_CAPAS, NUM_FILAS, NUM_COLUMNAS)

    influence_vector_1 = 1.1 - 2.2j
    influence_vector_2 = 5.0 + 0.5j
    influence_vector_3 = -1.0 + 10.0j

    campo_toroidal_global_servicio.aplicar_influencia(
        0, 0, 0, influence_vector_1, "test_vec_api_1"
    )
    campo_toroidal_global_servicio.aplicar_influencia(
        1, 2, 3, influence_vector_2, "test_vec_api_2"
    )
    # fmt: off
    campo_toroidal_global_servicio.aplicar_influencia(
        NUM_CAPAS - 1,
        NUM_FILAS - 1,
        NUM_COLUMNAS - 1,
        influence_vector_3,
        "test_vec_api_3"
    )
    # fmt: on

    response = cliente_flask.get("/api/ecu/field_vector")
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "success"
    assert "field_vector" in data
    assert "metadata" in data

    assert data["metadata"]["capas"] == NUM_CAPAS
    assert data["metadata"]["filas"] == NUM_FILAS
    assert data["metadata"]["columnas"] == NUM_COLUMNAS

    returned_field_list = data["field_vector"]
    assert isinstance(returned_field_list, list)
    assert len(returned_field_list) == NUM_CAPAS

    for capa_idx, layer_data in enumerate(returned_field_list):
        assert isinstance(layer_data, list)
        assert len(layer_data) == NUM_FILAS

        for row_idx, row_data in enumerate(layer_data):
            assert isinstance(row_data, list)
            assert len(row_data) == NUM_COLUMNAS

            for col_idx_loop, vector_data in enumerate(row_data):
                with campo_toroidal_global_servicio.lock:
                    actual_vector = (
                        campo_toroidal_global_servicio.campo_q[capa_idx]
                        [row_idx, col_idx_loop]
                    )
                assert np.isclose(
                    complex(vector_data[0], vector_data[1]), actual_vector
                )


def test_endpoint_ecu_error_interno(cliente_flask: FlaskClient, mocker):
    """Test: Manejo de errores internos en API /api/ecu (500)."""
    mocker.patch(
        'numpy.abs',
        side_effect=Exception("Mock error interno"))
    respuesta = cliente_flask.get("/api/ecu")
    assert respuesta.status_code == 500
    datos = respuesta.get_json()
    assert datos["status"] == "error"
    assert "error interno" in datos["message"].lower()


def test_set_initial_quantum_phase(campo_toroidal_test: ToroidalField):
    """Test: Inicialización de la fase cuántica a un estado aleatorio."""
    tf = campo_toroidal_test
    seed = 42
    tf.set_initial_quantum_phase(seed)

    # Verificar que todos los valores son números complejos con magnitud ~1
    for capa in tf.campo_q:
        magnitudes = np.abs(capa)
        assert np.allclose(magnitudes, 1.0)

    # Verificar que con la misma semilla, el resultado es idéntico
    tf2 = ToroidalField(tf.num_capas, tf.num_rows, tf.num_cols)
    tf2.set_initial_quantum_phase(seed)
    for capa1, capa2 in zip(tf.campo_q, tf2.campo_q):
        assert np.allclose(capa1, capa2)


def test_apply_quantum_step(campo_toroidal_test: ToroidalField):
    """Test: Aplicación de un paso de evolución cuántica."""
    tf = campo_toroidal_test
    tf.set_initial_quantum_phase(seed=123)
    initial_phases = [np.copy(capa) for capa in tf.campo_q]

    dt = 0.1
    tf.apply_quantum_step(dt)

    for i, initial_capa in enumerate(initial_phases):
        alpha = tf.alphas[i]
        expected_phase_change = np.exp(-1j * alpha * dt)
        expected_final_capa = initial_capa * expected_phase_change
        assert np.allclose(tf.campo_q[i], expected_final_capa)
