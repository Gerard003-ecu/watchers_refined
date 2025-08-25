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
    def validate_vector(
        vector_data: Any,
    ) -> Tuple[Optional[complex], Optional[str]]:
        """Valida y convierte el vector de influencia."""
        if not isinstance(vector_data, list) or len(vector_data) != 2:
            return (
                None,
                "El vector debe ser una lista de 2 elementos [real, imaginario]",
            )

        try:
            real_part = float(vector_data[0])
            imag_part = float(vector_data[1])
            return complex(real_part, imag_part), None
        except (ValueError, TypeError):
            return None, "Los componentes del vector deben ser números válidos"
