import schemathesis
from schemathesis import Case

# Cargar el esquema OpenAPI desde el archivo.
# La URL base se tomará de la sección `servers` del esquema.
schema = schemathesis.openapi.from_path("api-contracts/matriz_ecu.v1.yml")


@schema.parametrize()
def test_api_contract(case: Case):
    """
    Valida que la respuesta de la API se adhiere al contrato OpenAPI.
    """
    print(f"Testing endpoint: {case.method} {case.path}")

    # Realiza la llamada a la API con los datos generados por Schemathesis
    # y valida que la respuesta (código de estado, headers, cuerpo)
    # coincide con lo definido en el schema.
    # Se usa un base_url temporal para que el test se pueda ejecutar
    # en un entorno donde el servicio no está disponible.
    case.call_and_validate(base_url="http://localhost:8000")
