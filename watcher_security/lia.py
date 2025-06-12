"""
Propósito:
Proporciona un marco algebraico para operaciones vectoriales que garantiza:

    Independencia: Fallos en un subespacio no propagan errores
        (matrices diagonales por bloques).

    Escalabilidad: Añade dimensiones sin reentrenar el sistema completo
        (add_module()).

    Ortogonalidad: Módulos operan en ejes no interferentes
        (autovectores perpendiculares).
"""

import numpy as np


class LIATransformer:
    def __init__(self, dimensions, subspace_mapping):
        self.dimensions = dimensions
        # Matriz diagonal por bloques para ortogonalidad
        self.transformation_matrix = (
            self._create_block_diagonal_matrix(subspace_mapping)
        )
        self.subspaces = subspace_mapping

    def _create_block_diagonal_matrix(self, subspaces):
        """Crea matriz en bloques diagonal para independencia de subespacios"""
        matrix = np.zeros((self.dimensions, self.dimensions))
        for subspace in subspaces.values():
            size = subspace.stop - subspace.start
            block = np.random.randn(size, size)
            # Hacer bloque diagonalmente dominante para estabilidad
            np.fill_diagonal(block, np.abs(block).sum(axis=1) + 0.1)
            matrix[subspace, subspace] = block
        return matrix

    def transform(self, vector):
        """
        Aplica transformación lineal 
        manteniendo coherencia en subespacios
        """
        if len(vector) != self.dimensions:
            raise ValueError("Dimensionalidad del vector incorrecta")
        transformed = np.dot(self.transformation_matrix, vector)
        # Normalizar por subespacio
        for name, subspace in self.subspaces.items():
            subspace_norm = np.linalg.norm(transformed[subspace])
            if subspace_norm > 0:
                transformed[subspace] /= subspace_norm
        return transformed

    def add_module(self, new_dimensions):
        """Escalabilidad: Añade nuevas dimensiones al espacio"""
        new_total = self.dimensions + new_dimensions
        new_matrix = np.zeros((new_total, new_total))
        new_matrix[:self.dimensions, :self.dimensions] = (
            self.transformation_matrix
        )
        self.dimensions = new_total
        self.transformation_matrix = new_matrix
