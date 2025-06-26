# tests/integration/test_integration_malla_ecu.py

import pytest
import requests
import responses  # Para mockear el servidor ECU
import numpy as np
import json
import logging
# No necesitas Draft7Validator o SchemaError aquí si los esquemas ya están
# validados en ecu_schemas.py
from jsonschema import validate, ValidationError
from unittest.mock import patch  # Necesario para mockear módulos/funciones

# --- Importar Esquemas Compartidos ---
# ASUMIENDO que 'mi-proyecto' está en el PYTHONPATH o que pytest se ejecuta
# desde allí y que contracts/schemas/__init__.py existe.
from contracts.schemas.ecu_schemas import (
    ECU_FIELD_VECTOR_RESPONSE_SCHEMA,
    ECU_INFLUENCE_REQUEST_SCHEMA,
    ECU_INFLUENCE_RESPONSE_SCHEMA,
)

# --- Importaciones de los módulos bajo prueba ---
from watchers.watchers_tools.malla_watcher.malla_watcher import (
    fetch_and_apply_torus_field,
    send_influence_to_torus,
)
from watchers.watchers_tools.malla_watcher.utils.cilindro_grafenal import (
    HexCylindricalMesh,
    Cell,
)

# --- Configuración ---
logger = logging.getLogger(__name__)
ECU_MOCK_BASE_URL = "http://mock-ecu:8000"


# --- Fixtures ---
@pytest.fixture
def mock_http_server():
    """Activa el mock de responses para todas las llamadas HTTP en un test."""
    with responses.RequestsMock() as rsps:
        yield rsps


@pytest.fixture
def malla_instance_for_test():
    """Crea una instancia de malla simple para los tests que la necesiten."""
    malla = HexCylindricalMesh(
        radius=2,
        height_segments=2,
        circumference_segments_target=4,
        hex_size=1.0,
    )
    if not malla.cells:
        malla.cells[(0, 0)] = Cell(
            cyl_radius=2.0, cyl_theta=0.0, cyl_z=0.0, q_axial=0, r_axial=0
        )
    return malla


# --- Tests de Integración con Contratos ---


def test_malla_fetches_and_processes_ecu_field_vector(
    mock_http_server, malla_instance_for_test, caplog
):
    """
    Test Contrato: Malla Watcher (Consumidor) obtiene y procesa campo vectorial
    de Matriz ECU (Proveedor).
    1. Define respuesta mockeada de ECU que CUMPLE contrato
       ECU_FIELD_VECTOR_RESPONSE_SCHEMA.
    2. Llama a lógica de Malla Watcher para obtener y aplicar este campo.
    3. Verifica que Malla Watcher procesó datos sin errores.
    """
    caplog.set_level(logging.DEBUG)

    mock_num_capas, mock_num_filas, mock_num_cols = 1, 2, 2
    mock_field_data_payload = {
        "status": "success",
        "metadata": {
            "descripcion": "Campo mockeado para test",
            "capas": mock_num_capas,
            "filas": mock_num_filas,
            "columnas": mock_num_cols,
            "vector_dim": 2,
        },
        "field_vector": [  # Capa 0
            [  # Fila 0
                [0.1, 0.2],  # Col 0
                [0.3, 0.4],  # Col 1
            ],
            [  # Fila 1
                [0.5, 0.6],  # Col 0
                [0.7, 0.8],  # Col 1
            ],
        ],
    }
    # If mock_num_capas is 1, field_vector should be wrapped in another list
    if (
        mock_num_capas == 1
        and isinstance(
            mock_field_data_payload["field_vector"], list
        )
        and (
            len(mock_field_data_payload["field_vector"]) == mock_num_filas
        if mock_num_filas > 0
        else True
        )
    and (
        len(mock_field_data_payload["field_vector"][0]) == mock_num_cols
        if mock_num_filas > 0 and mock_num_cols > 0
        else True
        )
    ):
        mock_field_data_payload["field_vector"] = 
    [mock_field_data_payload["field_vector"]]

    validate(
        instance=mock_field_data_payload,
        schema=ECU_FIELD_VECTOR_RESPONSE_SCHEMA,
    )

    mock_http_server.add(
        responses.GET,
        f"{ECU_MOCK_BASE_URL}/api/ecu/field_vector",
        json=mock_field_data_payload,
        status=200,
    )

    # Mockear globales y apply_external_field_to_mesh si solo se testea
    # HTTP, no el procesamiento interno completo de malla_watcher.
    # Si se prueba el procesamiento, no mockear apply_external_field_to_mesh.
    patch_base_url = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "MATRIZ_ECU_BASE_URL",
        ECU_MOCK_BASE_URL,
    )
    patch_malla_global = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "malla_cilindrica_global",
        malla_instance_for_test,
    )
    patch_capas = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "TORUS_NUM_CAPAS",
        mock_num_capas,
    )
    patch_filas = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "TORUS_NUM_FILAS",
        mock_num_filas,
    )
    patch_columnas = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "TORUS_NUM_COLUMNAS",
        mock_num_cols,
    )

    with (
        patch_base_url,
        patch_malla_global,
        patch_capas,
        patch_filas,
        patch_columnas,
    ):
        # Opcional: mockear apply_external_field_to_mesh
        # with patch('watchers.watchers_tools.malla_watcher'
        #            '.malla_watcher.apply_external_field_to_mesh')
        #            as mock_apply:
        fetch_and_apply_torus_field()
        # if mockeado:
        #     mock_apply.assert_called_once_with(
        #         malla_instance_for_test,
        #         mock_field_data_payload["field_vector"]
        #     )

    assert len(mock_http_server.calls) == 1
    expected_url = f"{ECU_MOCK_BASE_URL}/api/ecu/field_vector"
    assert mock_http_server.calls[0].request.url == expected_url

    if len(malla_instance_for_test.cells) > 0:
        q_vectors_updated = sum(
            1
            for cell in malla_instance_for_test.get_all_cells()
            if not np.allclose(cell.q_vector, np.zeros(2))
        )
        assertion_msg = (
            "Ningún q_vector en la malla parece haber sido actualizado."
        )
        assert q_vectors_updated > 0, assertion_msg
        logger.info(f"{q_vectors_updated} q_vectors actualizados en la malla.")

    assert not any(
        record.levelno >= logging.ERROR
        for record in caplog.records
        if "malla_watcher" in record.name
    )


def test_malla_sends_valid_influence_to_ecu(mock_http_server, caplog):
    """
    Test Contrato: Malla Watcher (Consumidor) envía influencia a Matriz ECU
    (Proveedor). Verifica payload CUMPLE contrato y respuesta ECU procesada.
    """
    caplog.set_level(logging.DEBUG)
    dphi_dt_test_value = 7.89

    mock_ecu_response_payload = {
        "status": "success",
        "message": "Influencia aplicada.",
        "applied_to": {
            "capa": 0,
            "row": 1,
            "col": 2,
        },
        "vector": [dphi_dt_test_value, 0.0],
    }
    validate(
        instance=mock_ecu_response_payload,
        schema=ECU_INFLUENCE_RESPONSE_SCHEMA,
    )

    mock_http_server.add(
        responses.POST,
        f"{ECU_MOCK_BASE_URL}/api/ecu/influence",
        json=mock_ecu_response_payload,
        status=200,
    )

    test_torus_filas = 4  # Para que target_row sea 2
    test_torus_columnas = 6  # Para que target_col sea 3
    # Actualizar applied_to en mock_ecu_response_payload para que coincida
    mock_ecu_response_payload["applied_to"]["row"] = test_torus_filas // 2
    mock_ecu_response_payload["applied_to"]["col"] = test_torus_columnas // 2

    patch_base_url_inf = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "MATRIZ_ECU_BASE_URL",
        ECU_MOCK_BASE_URL,
    )
    patch_filas_inf = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "TORUS_NUM_FILAS",
        test_torus_filas,
    )
    patch_columnas_inf = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "TORUS_NUM_COLUMNAS",
        test_torus_columnas,
    )

    with (patch_base_url_inf, patch_filas_inf, patch_columnas_inf):
        send_influence_to_torus(dphi_dt_test_value)

    assert len(mock_http_server.calls) == 1
    sent_request = mock_http_server.calls[0].request
    assert sent_request.url == f"{ECU_MOCK_BASE_URL}/api/ecu/influence"

    sent_payload = json.loads(sent_request.body)
    try:
        validate(instance=sent_payload, schema=ECU_INFLUENCE_REQUEST_SCHEMA)
    except ValidationError as e:
        error_msg = (
            f"Payload de Malla Watcher NO cumple esquema: {e}\n"
            f"Payload: {sent_payload}"
        )
        pytest.fail(error_msg)

    assert sent_payload["vector"] == [dphi_dt_test_value, 0.0]
    assert sent_payload["row"] == test_torus_filas // 2
    assert sent_payload["col"] == test_torus_columnas // 2

    assert not any(
        record.levelno >= logging.ERROR
        for record in caplog.records
        if "malla_watcher" in record.name
    )


def test_ecu_accepts_valid_influence_payload(mock_http_server, caplog):
    """
    Test Contrato: Cliente envía influencia válida a Matriz ECU
    (Prov. mock). Verifica validez de payload y respuesta mock ECU.
    Enfocado en perspectiva del proveedor (ECU) al ser llamado.
    """
    caplog.set_level(logging.DEBUG)

    valid_influence_payload = {
        "capa": 0,
        "row": 1,
        "col": 2,
        "vector": [10.5, -3.3],
        "nombre_watcher": "integration_test_client",
    }
    # El cliente (este test) se asegura de enviar payload que cumple contrato
    validate(
        instance=valid_influence_payload,
        schema=ECU_INFLUENCE_REQUEST_SCHEMA
    )

    # ECU (mockeado) debe responder con algo que cumpla su contrato de
    # respuesta
    expected_ecu_response = {
        "status": "success",
        "message": "Influencia procesada por mock ECU.",
        "applied_to": {"capa": 0, "row": 1, "col": 2},
        "vector": [10.5, -3.3],
    }
    validate(
        instance=expected_ecu_response,
        schema=ECU_INFLUENCE_RESPONSE_SCHEMA
    )

    mock_http_server.add(  # Usar la fixture general mock_http_server
        responses.POST,
        f"{ECU_MOCK_BASE_URL}/api/ecu/influence",
        json=expected_ecu_response,
        status=200,
    )

    # Simular un cliente llamando al endpoint mockeado de ECU
    response = requests.post(
        f"{ECU_MOCK_BASE_URL}/api/ecu/influence",
        json=valid_influence_payload,
        timeout=5,
    )
    response.raise_for_status()

    response_data = response.json()
    try:
        validate(
            instance=response_data,
            schema=ECU_INFLUENCE_RESPONSE_SCHEMA
        )
    except ValidationError as e:
        error_msg = (
            f"Respuesta del mock ECU NO cumple esquema: {e}\n"
            f"Respuesta: {response_data}"
        )
        pytest.fail(error_msg)

    assert response_data["message"] == "Influencia procesada por mock ECU."


# --- Tests Adicionales (Manejo de Errores en la Interacción) ---
# (Estos tests permanecen prácticamente iguales, usando mock_http_server)


def test_malla_handles_ecu_field_vector_api_error(
    mock_http_server, malla_instance_for_test, caplog
):
    caplog.set_level(logging.INFO)
    mock_http_server.add(
        responses.GET,
        f"{ECU_MOCK_BASE_URL}/api/ecu/field_vector",
        json={"error": "ECU explotó"},
        status=500,
    )
    patch_matriz_url = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "MATRIZ_ECU_BASE_URL",
        ECU_MOCK_BASE_URL,
    )
    patch_malla = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher."
        "malla_cilindrica_global",
        malla_instance_for_test,
    )
    patch_apply = patch(
        "watchers.watchers_tools.malla_watcher.malla_watcher"
        ".apply_external_field_to_mesh"
    )
    with (patch_matriz_url, patch_malla, patch_apply as mock_apply_func):
        fetch_and_apply_torus_field()
        mock_apply_func.assert_not_called()
        log_message_found = any(
            "error de red o http al obtener campo vectorial"
            in rec.message.lower()
            for rec in caplog.records
            if "malla_watcher" in rec.name and rec.levelno >= logging.ERROR
        )
        assert log_message_found, (
            "Malla Watcher no logueó error apropiado "
            "tras fallo API ECU."
        )


def test_malla_handles_ecu_influence_api_error(mock_http_server, caplog):
    caplog.set_level(logging.INFO)
    dphi_dt_test_value = 3.14
    mock_http_server.add(
        responses.POST,
        f"{ECU_MOCK_BASE_URL}/api/ecu/influence",
        json={"error": "Payload inválido según ECU"},
        status=400,
    )
    with (
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher."
            "MATRIZ_ECU_BASE_URL",
            ECU_MOCK_BASE_URL,
        ),
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher."
            "TORUS_NUM_FILAS",
            4,
        ),
        patch(
            "watchers.watchers_tools.malla_watcher.malla_watcher."
            "TORUS_NUM_COLUMNAS",
            6,
        ),
    ):
        send_influence_to_torus(dphi_dt_test_value)
    assert any(
        "error de red al enviar influencia" in rec.message.lower()
        for rec in caplog.records
        if (
            ("malla_watcher" in rec.name) and
            (rec.levelno >= logging.ERROR)
        )  # Closing parenthesis for 'if'
    )  # Closing parenthesis for 'any'
