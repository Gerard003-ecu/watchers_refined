# --- START OF FILE test_matriz_ecu.py (CORREGIDO) ---

# tests/unit/test_matriz_ecu.py

import pytest
import numpy as np
import time
import threading
from ecu.matriz_ecu import (
    ToroidalField,
    app,
    campo_toroidal_global_servicio,
    NUM_CAPAS, NUM_FILAS, NUM_COLUMNAS, # Importar constantes globales para tests API
    stop_simulation_event # Importar si se usa (no parece usarse en estos tests, pero mantener si es necesario)
)
from flask.testing import FlaskClient
import logging
from unittest.mock import MagicMock # Importar MagicMock si se usa (no parece usarse directamente aquí, pero mantener si es necesario)
import unittest.mock # Importar unittest.mock para patch

logger = logging.getLogger(__name__)

# --- Fixtures ---
@pytest.fixture
def campo_toroidal_test():
    """Fixture: Campo toroidal aislado para tests unitarios de la clase."""
    # Usar dimensiones diferentes a las globales para asegurar aislamiento
    return ToroidalField(num_capas=2, num_rows=3, num_cols=4, alphas=None, dampings=None)

@pytest.fixture
def cliente_flask() -> FlaskClient:
    """Fixture: Cliente de pruebas de Flask."""
    app.config['TESTING'] = True
    with app.test_client() as cliente:
        # Resetear estado global antes de cada test API para predictibilidad
        # (Opcional pero recomendado si los tests modifican el estado global)
        # Usar la instancia global importada
        with campo_toroidal_global_servicio.lock:
             # Usar las dimensiones globales correctas para la instancia global
             campo_toroidal_global_servicio.campo = [
                 np.zeros((NUM_FILAS, NUM_COLUMNAS, 2)) for _ in range(NUM_CAPAS)
             ]
             # Resetear alphas y dampings a defaults si es necesario para consistencia
             # campo_toroidal_global_servicio.alphas = [...]
             # campo_toroidal_global_servicio.dampings = [...]

        logger.debug("Estado del campo global reseteado para test API")
        yield cliente

# --- Tests para ToroidalField (Usando instancia aislada 'campo_toroidal_test') ---

def test_inicializacion_valida(campo_toroidal_test: ToroidalField):
    """Test: Creación correcta con parámetros válidos (instancia aislada)."""
    tf = campo_toroidal_test
    assert tf.num_capas == 2
    assert tf.num_rows == 3
    assert tf.num_cols == 4
    assert isinstance(tf.campo, list)
    assert len(tf.campo) == 2
    assert tf.campo[0].shape == (3, 4, 2)
    assert np.all(tf.campo[0] == 0)
    # Verificar que alphas y dampings tienen la longitud correcta (usando defaults)
    assert len(tf.alphas) == 2
    assert len(tf.dampings) == 2
    assert isinstance(tf.lock, type(threading.Lock()))

def test_inicializacion_con_params_capa():
    """Test: Creación correcta pasando alphas y dampings."""
    alphas_test = [0.1, 0.2]
    dampings_test = [0.01, 0.02]
    tf = ToroidalField(num_capas=2, num_rows=2, num_cols=2, alphas=alphas_test, dampings=dampings_test)
    assert tf.alphas == alphas_test
    assert tf.dampings == dampings_test

def test_inicializacion_params_capa_longitud_incorrecta():
    """Test: Error al crear con listas de params de longitud incorrecta."""
    with pytest.raises(ValueError, match="La lista 'alphas' debe tener longitud 2"):
        ToroidalField(num_capas=2, num_rows=2, num_cols=2, alphas=[0.1])
    with pytest.raises(ValueError, match="La lista 'dampings' debe tener longitud 3"):
        ToroidalField(num_capas=3, num_rows=2, num_cols=2, dampings=[0.1, 0.2])

def test_inicializacion_invalida():
    """Test: Error al crear con dimensiones inválidas."""
    with pytest.raises(ValueError, match="dimensiones .* deben ser positivas"):
        ToroidalField(num_capas=0, num_rows=2, num_cols=2)
    with pytest.raises(ValueError, match="dimensiones .* deben ser positivas"):
        ToroidalField(num_capas=1, num_rows=-1, num_cols=2)
    with pytest.raises(ValueError, match="dimensiones .* deben ser positivas"):
        ToroidalField(num_capas=1, num_rows=2, num_cols=0)

def test_aplicar_influencia_valida(campo_toroidal_test: ToroidalField):
    """Test: Aplicar influencia (inyección vectorial) en posición válida (instancia aislada)."""
    tf = campo_toroidal_test
    capa, row, col = 0, 1, 2
    vector = np.array([0.5, -0.3])
    with tf.lock:
        valor_inicial = np.copy(tf.campo[capa][row, col])

    success = tf.aplicar_influencia(capa, row, col, vector, "test_watcher")
    assert success is True

    with tf.lock:
        valor_final = tf.campo[capa][row, col]
    assert np.array_equal(valor_final, valor_inicial + vector)

def test_aplicar_influencia_fuera_rango(campo_toroidal_test: ToroidalField, caplog):
    """Test: Manejo de índices fuera de rango al aplicar influencia (instancia aislada)."""
    tf = campo_toroidal_test
    vector = np.array([1.0, 0.0])
    with tf.lock:
        valor_original = [np.copy(c) for c in tf.campo]

    with caplog.at_level(logging.ERROR):
        success = tf.aplicar_influencia(5, 1, 1, vector, "watcher_err_capa")
    assert success is False
    assert "índice de capa fuera de rango (5)" in caplog.text.lower()
    with tf.lock:
        for i in range(tf.num_capas):
            assert np.array_equal(tf.campo[i], valor_original[i])
    caplog.clear()

    with caplog.at_level(logging.ERROR):
         success = tf.aplicar_influencia(0, 5, 2, vector, "watcher_err_fila")
    assert success is False
    assert "índice de fila fuera de rango (5)" in caplog.text.lower()
    with tf.lock:
        for i in range(tf.num_capas):
            assert np.array_equal(tf.campo[i], valor_original[i])
    caplog.clear()

    with caplog.at_level(logging.ERROR):
         success = tf.aplicar_influencia(0, 1, -1, vector, "watcher_err_col")
    assert success is False
    assert "índice de columna fuera de rango (-1)" in caplog.text.lower()
    with tf.lock:
        for i in range(tf.num_capas):
            assert np.array_equal(tf.campo[i], valor_original[i])

def test_aplicar_influencia_vector_invalido(campo_toroidal_test: ToroidalField, caplog):
    """Test: Aplicar influencia con vector de formato incorrecto."""
    tf = campo_toroidal_test
    with tf.lock:
        valor_original = [np.copy(c) for c in tf.campo]
    with caplog.at_level(logging.ERROR):
        success = tf.aplicar_influencia(0, 1, 1, np.array([1, 2, 3]), "watcher_vec_err")
    assert success is False
    assert "vector de influencia inválido" in caplog.text.lower()
    assert "NumPy array (2,)" in caplog.text
    with tf.lock:
        for i in range(tf.num_capas):
            assert np.array_equal(tf.campo[i], valor_original[i])

def test_get_neighbors_conectividad_toroidal(campo_toroidal_test: ToroidalField):
    """Test: Vecinos con wraparound toroidal (instancia aislada)."""
    tf = campo_toroidal_test # 2 capas, 3x4
    vecinos_00 = tf.get_neighbors(0, 0)
    # Expected neighbors for (row=0, col=0) in a 3x4 grid:
    # Up: (0-1)%3 = 2 -> (2, 0)
    # Down: (0+1)%3 = 1 -> (1, 0)
    # Left: (0-1)%4 = 3 -> (0, 3)
    # Right: (0+1)%4 = 1 -> (0, 1)
    expected_00 = [(2,0), (1,0), (0,3), (0,1)]
    assert set(vecinos_00) == set(expected_00), "Error en vecinos de (0,0)"

    # Expected neighbors for (row=2, col=3) in a 3x4 grid:
    # Up: (2-1)%3 = 1 -> (1, 3)
    # Down: (2+1)%3 = 0 -> (0, 3)
    # Left: (3-1)%4 = 2 -> (2, 2)
    # Right: (3+1)%4 = 0 -> (2, 0)
    vecinos_23 = tf.get_neighbors(2, 3)
    expected_23 = [(1,3), (0,3), (2,2), (2,0)]
    assert set(vecinos_23) == set(expected_23), "Error en vecinos de (2,3)"

def test_calcular_gradiente_adaptativo(campo_toroidal_test: ToroidalField):
    """Test: Cálculo de gradiente (diferencia de magnitud entre capas) (instancia aislada)."""
    tf = campo_toroidal_test # 2 capas, 3x4
    # Aplicar vectores para crear una diferencia entre capa 0 y capa 1 en (1,1)
    tf.aplicar_influencia(0, 1, 1, np.array([3.0, 4.0]), "test_grad1_capa0") # Mag 5 en capa 0
    tf.aplicar_influencia(1, 1, 1, np.array([0.0, -6.0]), "test_grad1_capa1") # Mag 6 en capa 1

    gradiente = tf.calcular_gradiente_adaptativo()
    # Shape esperado: (num_capas - 1, num_rows, num_cols)
    assert gradiente.shape == (1, 3, 4)

    # Diferencia vectorial en (0, 1, 1) = campo[0][1,1] - campo[1][1,1]
    # = [3.0, 4.0] - [0.0, -6.0] = [3.0, 10.0]
    # Magnitud de la diferencia = sqrt(3.0^2 + 10.0^2) = sqrt(9 + 100) = sqrt(109)
    expected_diff_mag = np.linalg.norm(np.array([3.0, 10.0])) # sqrt(109)
    assert gradiente[0, 1, 1] == pytest.approx(expected_diff_mag)

    # En (0,0,0), ambos campos son cero, la diferencia es cero
    assert gradiente[0, 0, 0] == pytest.approx(0.0)

def test_calcular_gradiente_sin_suficientes_capas():
    """Test: Gradiente con menos de 2 capas."""
    tf = ToroidalField(num_capas=1, num_rows=2, num_cols=2)
    gradiente = tf.calcular_gradiente_adaptativo()
    assert gradiente.size == 0
    assert gradiente.shape == (0, ) # Verificar shape correcto (0 capas de gradiente)

def test_obtener_campo_unificado(campo_toroidal_test: ToroidalField):
    """Test: Campo unificado (intensidad agregada ponderada) con pesos por capa (instancia aislada)."""
    tf = campo_toroidal_test # 2 capas, 3x4
    # Aplicar vectores en la misma posición (1,1) en ambas capas
    tf.aplicar_influencia(0, 1, 1, np.array([3.0, 4.0]), "test_uni1_capa0") # Mag 5 en capa 0
    tf.aplicar_influencia(1, 1, 1, np.array([0.0, -6.0]), "test_uni2_capa1") # Mag 6 en capa 1

    campo_uni = tf.obtener_campo_unificado()
    assert campo_uni.shape == (3, 4)

    # Pesos por defecto para 2 capas: [1.0, 0.5]
    # Valor esperado en (1,1) = peso_capa_0 * Mag_capa_0 + peso_capa_1 * Mag_capa_1
    expected_value_11 = 1.0 * 5.0 + 0.5 * 6.0 # 5.0 + 3.0 = 8.0
    assert campo_uni[1, 1] == pytest.approx(expected_value_11)

    # En (0,0), ambos campos son cero, la magnitud unificada es cero
    assert campo_uni[0, 0] == pytest.approx(0.0)

def test_obtener_campo_unificado_una_capa():
    """Test: Campo unificado con una sola capa."""
    tf = ToroidalField(num_capas=1, num_rows=2, num_cols=2)
    tf.aplicar_influencia(0, 0, 0, np.array([1.0, 1.0]), "test_uni_1capa") # Mag sqrt(2)
    campo_uni = tf.obtener_campo_unificado()
    assert campo_uni.shape == (2, 2)
    # Con una capa, el peso es 1.0
    assert campo_uni[0, 0] == pytest.approx(1.0 * np.sqrt(2.0))
    assert campo_uni[0, 1] == pytest.approx(0.0) # Otros puntos son cero

def test_apply_rotational_step(campo_toroidal_test: ToroidalField):
    """Test: Aplicación de un paso rotacional (advección, acoplamiento, disipación) (instancia aislada)."""
    tf = campo_toroidal_test # 2 capas, 3x4
    initial_vector = np.array([1.0, 2.0])
    capa, row, col = 0, 1, 1 # Aplicar influencia en capa 0, (1,1)
    tf.aplicar_influencia(capa, row, col, initial_vector, "test_rot_watcher")

    dt, beta = 0.1, 0.1

    # Capturar estado inicial completo para verificar que solo cambian los afectados
    with tf.lock:
        initial_state_full = [np.copy(c) for c in tf.campo]

    tf.apply_rotational_step(dt, beta)

    with tf.lock:
        final_state_full = tf.campo

    # Verificar que el nodo original (0, 1, 1) ha cambiado
    assert not np.allclose(final_state_full[capa][row, col], initial_state_full[capa][row, col]), "Nodo original debería haber cambiado"
    assert np.all(np.isfinite(final_state_full[capa][row, col])), "Valores no finitos en nodo final"

    # Verificar que los vecinos afectados en la misma capa (0, 1, 0) y (0, 1, 2) han cambiado (advección)
    # y los vecinos afectados en otras filas (0, 0, 1) y (0, 2, 1) han cambiado (acoplamiento vertical)
    # y los vecinos afectados en otras capas (1, 1, 1) han cambiado (acoplamiento vertical)

    # Coordenadas de los nodos que deberían cambiar debido a la influencia en (0, 1, 1)
    # En capa 0: (1,1) (original), (1,0) (left), (0,1) (up), (2,1) (down)
    # En capa 1: (1,1) (acoplamiento vertical desde capa 0)
    nodes_that_should_change = [
        (0, 1, 1), # Nodo original (afectado por disipación y vecinos)
        (0, 1, 2), # Vecino derecho (afectado por su vecino izquierdo (0,1,1))
        (0, 0, 1), # Vecino superior (afectado por su vecino inferior (0,1,1))
        (0, 2, 1), # Vecino inferior (afectado por su vecino superior (0,1,1))
    ]

    for c in range(tf.num_capas):
        for r in range(tf.num_rows):
            for col in range(tf.num_cols):
                coords = (c, r, col)
                if coords in nodes_that_should_change:
                     # Estos nodos deberían haber cambiado
                     assert not np.allclose(final_state_full[c][r, col], initial_state_full[c][r, col]), f"Nodo {coords} debería haber cambiado"
                else:
                     # Estos nodos NO deberían haber cambiado
                     assert np.allclose(final_state_full[c][r, col], initial_state_full[c][r, col]), f"Nodo {coords} NO debería haber cambiado"

# --- Tests para la API REST (Usando instancia global y cliente_flask) ---

def test_endpoint_health(cliente_flask: FlaskClient):
    """Test: Endpoint /api/health (esperando simulación inactiva)."""
    # La fixture cliente_flask resetea el campo global a ceros.
    # La simulación no se inicia automáticamente en el entorno de test.
    respuesta = cliente_flask.get("/api/health")
    datos = respuesta.get_json()
    # Esperamos status "warning" o "error" porque la simulación no está corriendo
    # Y el campo está reseteado a cero (podría considerarse error si 0 celdas es error)
    # Basado en la lógica de health_check en malla_watcher, si num_cells=0, status="error", code=500/503
    # Aquí en matriz_ecu, si field_initialized=True (que lo es por la instancia global)
    # y simulation_running=False, el status es "warning", code=503.
    assert respuesta.status_code == 503 # Esperar 503 si simulación inactiva
    assert datos["status"] == "warning" # Esperar status warning si simulación inactiva
    assert datos["field_initialized"] is True # La instancia global existe
    assert datos["simulation_running"] is False # El hilo no se inicia en tests

def test_endpoint_ecu_api(cliente_flask: FlaskClient):
    """Test: Respuesta exitosa del endpoint /api/ecu (instancia global)."""
    vector_influencia = np.array([1.1, -2.2])
    # Usar la instancia global importada
    campo_toroidal_global_servicio.aplicar_influencia(0, 0, 0, vector_influencia, "test_api_ecu")

    respuesta = cliente_flask.get("/api/ecu")
    assert respuesta.status_code == 200
    datos = respuesta.get_json()
    assert datos["status"] == "success"
    assert "estado_campo_unificado" in datos
    assert isinstance(datos["estado_campo_unificado"], list)
    # Usar las constantes globales importadas para verificar metadatos
    assert datos["metadata"]["capas"] == NUM_CAPAS
    assert datos["metadata"]["filas"] == NUM_FILAS
    assert datos["metadata"]["columnas"] == NUM_COLUMNAS

    # Verificar el valor del campo unificado en el punto donde se aplicó la influencia
    with campo_toroidal_global_servicio.lock:
        valor_nodo_000 = campo_toroidal_global_servicio.campo[0][0, 0]
    norma_esperada_000 = np.linalg.norm(valor_nodo_000)
    # Calcular pesos basados en NUM_CAPAS global (debe coincidir con la lógica de obtener_campo_unificado)
    pesos = np.linspace(1.0, 0.5, NUM_CAPAS) if NUM_CAPAS > 1 else np.array([1.0])
    peso_capa_0 = pesos[0]
    # Asumiendo que las otras capas están en 0 (debido al reset en la fixture)
    # El valor en estado_campo_unificado[0][0] es la suma ponderada de las magnitudes en (0,0) a través de todas las capas
    # Como solo aplicamos influencia en capa 0, solo esa capa contribuye.
    assert datos["estado_campo_unificado"][0][0] == pytest.approx(peso_capa_0 * norma_esperada_000)

def test_endpoint_influence_valido(cliente_flask: FlaskClient):
    """Test: Aplicar influencia válida vía /api/ecu/influence."""
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
    vector_np = np.array(payload["vector"])

    with campo_toroidal_global_servicio.lock:
        valor_inicial = np.copy(campo_toroidal_global_servicio.campo[capa_idx][row_idx, col_idx])

    respuesta = cliente_flask.post("/api/ecu/influence", json=payload)
    assert respuesta.status_code == 200
    datos = respuesta.get_json()
    assert datos["status"] == "success"
    assert datos["message"] == f"Influencia de '{payload['nombre_watcher']}' aplicada." # Verificar mensaje
    assert datos["applied_to"]["capa"] == capa_idx
    assert datos["applied_to"]["row"] == row_idx
    assert datos["applied_to"]["col"] == col_idx
    assert datos["vector"] == payload["vector"] # Comparar con lista original

    with campo_toroidal_global_servicio.lock:
        valor_final = campo_toroidal_global_servicio.campo[capa_idx][row_idx, col_idx]
    assert np.array_equal(valor_final, valor_inicial + vector_np)

def test_endpoint_influence_invalido_datos(cliente_flask: FlaskClient):
    """Test: Errores 400 por datos inválidos en /api/ecu/influence."""
    # Payload vacío
    respuesta = cliente_flask.post("/api/ecu/influence", json={})
    assert respuesta.status_code == 400
    # Verificar mensaje correcto para payload vacío
    assert "payload json vacío o ausente" in respuesta.get_json()["message"].lower()

    # Payload con campos faltantes
    payload_faltan = {"capa": 0, "row": 1}
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_faltan)
    assert respuesta.status_code == 400
    assert "faltan campos requeridos" in respuesta.get_json()["message"].lower()

    # Tipo incorrecto
    payload_tipo_err = {"capa": "cero", "row": 1, "col": 1, "vector": [1,1], "nombre_watcher": "t"}
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_tipo_err)
    assert respuesta.status_code == 400
    assert "errores de tipo" in respuesta.get_json()["message"].lower()
    assert "campo 'capa' debe ser int, recibido str" in respuesta.get_json()["message"].lower()

    # Vector incorrecto (lista de longitud incorrecta)
    payload_vec_err_len = {"capa": 0, "row": 1, "col": 1, "vector": [1], "nombre_watcher": "t"}
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_vec_err_len)
    assert respuesta.status_code == 400
    assert "errores de tipo" in respuesta.get_json()["message"].lower()
    assert "lista de 2 números" in respuesta.get_json()["message"].lower()

    # Vector incorrecto (contiene no números)
    payload_vec_err_type = {"capa": 0, "row": 1, "col": 1, "vector": [1, "a"], "nombre_watcher": "t"}
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_vec_err_type)
    assert respuesta.status_code == 400
    assert "errores de tipo" in respuesta.get_json()["message"].lower()
    assert "campo 'vector' debe contener números" in respuesta.get_json()["message"].lower()

def test_endpoint_influence_invalido_rango(cliente_flask: FlaskClient):
    """Test: Error 400 por índices fuera de rango en /api/ecu/influence."""
    # Usar constantes globales para asegurar fuera de rango
    payload_capa = {
        "capa": NUM_CAPAS, # Índice fuera de rango
        "row": 0, "col": 0, "vector": [1.0, 0.0], "nombre_watcher": "api_test_range_capa"
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_capa)
    assert respuesta.status_code == 400
    assert "error de validación al aplicar influencia" in respuesta.get_json()["message"].lower()

    payload_row = {
        "capa": 0,
        "row": NUM_FILAS, # Índice fuera de rango
        "col": 0, "vector": [1.0, 0.0], "nombre_watcher": "api_test_range_row"
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_row)
    assert respuesta.status_code == 400
    assert "error de validación al aplicar influencia" in respuesta.get_json()["message"].lower()

    payload_col = {
        "capa": 0, "row": 0,
        "col": NUM_COLUMNAS, # Índice fuera de rango
        "vector": [1.0, 0.0], "nombre_watcher": "api_test_range_col"
    }
    respuesta = cliente_flask.post("/api/ecu/influence", json=payload_col)
    assert respuesta.status_code == 400
    assert "error de validación al aplicar influencia" in respuesta.get_json()["message"].lower()

def test_endpoint_get_field_vector(cliente_flask):
    """
    Test: Endpoint /api/ecu/field_vector retorna el campo vectorial completo.
    """
    # cliente_flask fixture resets the global campo_toroidal_global_servicio
    # to zeros and provides the test client.

    # Apply some known influence to the global field instance
    # to ensure it's not all zeros and we can verify values.
    # Use the global instance imported from matriz_ecu
    from ecu.matriz_ecu import campo_toroidal_global_servicio, NUM_CAPAS, NUM_FILAS, NUM_COLUMNAS

    influence_vector_1 = np.array([1.1, -2.2])
    influence_vector_2 = np.array([5.0, 0.5])
    influence_vector_3 = np.array([-1.0, 10.0])

    # Apply influence at different locations
    campo_toroidal_global_servicio.aplicar_influencia(0, 0, 0, influence_vector_1, "test_vec_api_1")
    campo_toroidal_global_servicio.aplicar_influencia(1, 2, 3, influence_vector_2, "test_vec_api_2")
    campo_toroidal_global_servicio.aplicar_influencia(NUM_CAPAS - 1, NUM_FILAS - 1, NUM_COLUMNAS - 1, influence_vector_3, "test_vec_api_3")

    # Action: Make the GET request to the new endpoint
    response = cliente_flask.get("/api/ecu/field_vector")

    # Verification
    assert response.status_code == 200
    data = response.get_json()

    assert data["status"] == "success"
    assert "field_vector" in data
    assert "metadata" in data

    # Verify metadata
    assert data["metadata"]["capas"] == NUM_CAPAS
    assert data["metadata"]["filas"] == NUM_FILAS
    assert data["metadata"]["columnas"] == NUM_COLUMNAS
    assert data["metadata"]["vector_dim"] == 2
    assert "descripcion" in data["metadata"] # Check for description key

    # Verify the structure and values of the returned field_vector
    returned_field_list = data["field_vector"]
    assert isinstance(returned_field_list, list)
    assert len(returned_field_list) == NUM_CAPAS # Check number of layers

    for capa_idx, layer_data in enumerate(returned_field_list):
        assert isinstance(layer_data, list)
        assert len(layer_data) == NUM_FILAS # Check number of rows per layer

        for row_idx, row_data in enumerate(layer_data):
            assert isinstance(row_data, list)
            assert len(row_data) == NUM_COLUMNAS # Check number of columns per row

            for col_idx, vector_data in enumerate(row_data):
                assert isinstance(vector_data, list)
                assert len(vector_data) == 2 # Check vector dimension

                # Compare the returned vector with the actual state in the global instance
                # Access the global field state using the lock
                with campo_toroidal_global_servicio.lock:
                    actual_vector = campo_toroidal_global_servicio.campo[capa_idx][row_idx, col_idx]

                # Use np.allclose for floating-point comparison
                assert np.allclose(np.array(vector_data), actual_vector), \
                    f"Vector mismatch at ({capa_idx}, {row_idx}, {col_idx}): Returned {vector_data}, Expected {actual_vector}"

def test_endpoint_ecu_error_interno(cliente_flask: FlaskClient, mocker):
    """Test: Manejo de errores internos en el endpoint /api/ecu."""
    # Mockear el método interno que se llama para simular un error
    mocker.patch('ecu.matriz_ecu.ToroidalField.obtener_campo_unificado', # Patch en la CLASE, no en la instancia global
                 side_effect=Exception("Mock error interno"))
    respuesta = cliente_flask.get("/api/ecu")
    assert respuesta.status_code == 500
    datos = respuesta.get_json()
    assert datos["status"] == "error"
    assert "error interno" in datos["message"].lower()

# --- END OF FILE test_matriz_ecu.py (CORREGIDO) ---