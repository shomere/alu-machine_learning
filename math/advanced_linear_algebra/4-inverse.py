#!/usr/bin/env python3
"""
Matrix Inverse Calculation Module

This module provides a function to calculate the inverse
of a square matrix using the adjugate method.
"""


def inverse(matrix):
    """
    Calculate the inverse of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix

    Returns:
        list: The inverse matrix, or None if matrix is singular

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not a non-empty square matrix
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is a list of lists
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Base case: 1x1 matrix
    if n == 1:
        if matrix[0][0] == 0:
            return None
        return [[1 / matrix[0][0]]]

    # Calculate determinant first
    # Helper function for determinant calculation
    def det_2x2(mat):
        if len(mat) == 2:
            return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
        return mat[0][0]

    def calculate_determinant(mat):
        size = len(mat)
        if size == 1:
            return mat[0][0]
        if size == 2:
            return det_2x2(mat)
        det = 0
        for col in range(size):
            # Create submatrix
            sub = []
            for r in range(1, size):
                subrow = []
                for c in range(size):
                    if c != col:
                        subrow.append(mat[r][c])
                sub.append(subrow)
            sign = 1 if col % 2 == 0 else -1
            det += sign * mat[0][col] * calculate_determinant(sub)
        return det

    determinant_val = calculate_determinant(matrix)

    # Check if matrix is singular (determinant = 0)
    if determinant_val == 0:
        return None

    # Calculate adjugate matrix
    # First calculate cofactor matrix
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            # Create submatrix by removing row i and column j
            submatrix = []
            for row_idx in range(n):
                if row_idx != i:
                    subrow = []
                    for col_idx in range(n):
                        if col_idx != j:
                            subrow.append(matrix[row_idx][col_idx])
                    submatrix.append(subrow)
            # Calculate determinant of submatrix
            minor_ij = calculate_determinant(submatrix)
            # Apply sign based on position: (-1)^(i+j)
            sign = 1 if (i + j) % 2 == 0 else -1
            cofactor_ij = sign * minor_ij
            cofactor_row.append(cofactor_ij)
        cofactor_matrix.append(cofactor_row)

    # Transpose the cofactor matrix to get adjugate
    adjugate_matrix = []
    for i in range(n):
        adjugate_row = []
        for j in range(n):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    # Calculate inverse: (1/determinant) * adjugate
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adjugate_matrix[i][j] / determinant_val)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
