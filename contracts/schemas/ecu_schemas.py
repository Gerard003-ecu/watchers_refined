# mi-proyecto/contracts/schemas/ecu_schemas.py
from jsonschema import Draft7Validator, SchemaError
import logging

logger = logging.getLogger(__name__)

# Contrato para la respuesta de: GET /api/ecu/field_vector
ECU_FIELD_VECTOR_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success"]},
        "metadata": {
            "type": "object",
            "properties": {
                "descripcion": {"type": "string"},
                "capas": {"type": "integer", "minimum": 0},
                "filas": {"type": "integer", "minimum": 0},
                "columnas": {"type": "integer", "minimum": 0},
                "vector_dim": {"type": "integer", "const": 2}
            },
            "required": ["descripcion", "capas", "filas", "columnas", "vector_dim"]
        },
        "field_vector": {
            "type": "array",
            "items": { # Capas
                "type": "array",
                "items": { # Filas
                    "type": "array",
                    "items": { # Columnas
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    }
                }
            }
        }
    },
    "required": ["status", "metadata", "field_vector"]
}

# Contrato para el payload de solicitud de: POST /api/ecu/influence
ECU_INFLUENCE_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "capa": {"type": "integer", "minimum": 0},
        "row": {"type": "integer", "minimum": 0},
        "col": {"type": "integer", "minimum": 0},
        "vector": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
        },
        "nombre_watcher": {"type": "string", "minLength": 1}
    },
    "required": ["capa", "row", "col", "vector", "nombre_watcher"]
}

# Contrato para la respuesta esperada de: POST /api/ecu/influence (éxito)
ECU_INFLUENCE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success"]},
        "message": {"type": "string"},
        "applied_to": {
            "type": "object",
            "properties": {
                "capa": {"type": "integer"},
                "row": {"type": "integer"},
                "col": {"type": "integer"}
            },
            "required": ["capa", "row", "col"]
        },
        "vector": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2
        }
    },
    "required": ["status", "message", "applied_to", "vector"]
}

# Validar los esquemas al definir el módulo
schemas_to_validate = {
    "ECU_FIELD_VECTOR_RESPONSE_SCHEMA": ECU_FIELD_VECTOR_RESPONSE_SCHEMA,
    "ECU_INFLUENCE_REQUEST_SCHEMA": ECU_INFLUENCE_REQUEST_SCHEMA,
    "ECU_INFLUENCE_RESPONSE_SCHEMA": ECU_INFLUENCE_RESPONSE_SCHEMA,
}

for name, schema in schemas_to_validate.items():
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as e:
        logger.critical(f"Esquema {name} es inválido: {e}")
        # Podrías querer que esto falle ruidosamente si un esquema es incorrecto
        raise RuntimeError(f"Esquema de contrato {name} inválido al cargar el módulo de esquemas.") from e