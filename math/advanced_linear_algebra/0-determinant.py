#!/usr/bin/env python3
"""
Determinant Calculation Module

This module provides a function to calculate the determinant
of a square matrix using recursion (Laplace expansion).
"""


def determinant(matrix):
    """
    Calculate the determinant of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix

    Returns:
        float/int: The determinant of the matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    # Check if matrix is a list
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    # Check if matrix is a list of lists
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix case
    if matrix == [[]]:
        return 1

    # Check if matrix is square
    n = len(matrix)
    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a square matrix")

    # Base cases
    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive case for larger matrices
    det = 0
    for col in range(n):
        # Create submatrix by removing first row and current column
        submatrix = []
        for i in range(1, n):
            subrow = []
            for j in range(n):
                if j != col:
                    subrow.append(matrix[i][j])
            submatrix.append(subrow)

        # Calculate cofactor and add to determinant
        cofactor = matrix[0][col]
        sign = 1 if col % 2 == 0 else -1
        det += sign * cofactor * determinant(submatrix)

    return det
