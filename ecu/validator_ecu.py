"""Módulo para la validación de datos en la ECU."""

from typing import Any, List, Optional, Tuple


class InfluenceValidator:
    """Clase dedicada a validar influencias con métodos más específicos."""

    @staticmethod
    def validate_coordinates(capa: int, row: int, col: int, field) -> List[str]:
        """Valida que las coordenadas estén dentro del rango permitido."""
        errors = []
        if not (0 <= capa < field.num_capas):
            errors.append(
                f"Índice de capa {capa} fuera de rango [0, {field.num_capas - 1}]"
            )
        if not (0 <= row < field.num_rows):
            errors.append(
                f"Índice de fila {row} fuera de rango [0, {field.num_rows - 1}]"
            )
        if not (0 <= col < field.num_cols):
            errors.append(
                f"Índice de columna {col} fuera de rango [0, {field.num_cols - 1}]"
            )
        return errors

    @staticmethod
    def validate_vector(vec: Any) -> Tuple[Optional[complex], Optional[str]]:
        """Valida y convierte el vector de influencia desde múltiples formatos."""
        if isinstance(vec, list) and len(vec) == 2:
            try:
                return complex(float(vec[0]), float(vec[1])), None
            except (ValueError, TypeError):
                return None, "Vector debe ser [real, imag] con números válidos."
        elif isinstance(vec, dict) and "real" in vec and "imag" in vec:
            try:
                return complex(float(vec["real"]), float(vec["imag"])), None
            except (ValueError, TypeError) as e:
                return (
                    None,
                    f"Vector debe tener claves 'real' e 'imag' numéricas. Error: {e}",
                )
        else:
            return (
                None,
                "Formato de vector inválido. Use [real, imag] o {'real': ..., 'imag': ...}.",
            )
