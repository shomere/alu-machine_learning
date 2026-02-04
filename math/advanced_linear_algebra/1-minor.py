#!/usr/bin/env python3
"""
Minor Matrix Calculation Module

This module provides a function to calculate the minor matrix
of a square matrix.
"""


def minor(matrix):
    """
    Calculate the minor matrix of a square matrix.

    Args:
        matrix: A list of lists representing a square matrix

    Returns:
        list: The minor matrix

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
        return [[1]]

    # Calculate minor matrix
    minor_matrix = []
    for i in range(n):
        minor_row = []
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
            # Helper function for determinant calculation
            def calculate_determinant(mat):
                size = len(mat)
                if size == 1:
                    return mat[0][0]
                if size == 2:
                    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
                det = 0
                for col in range(size):
                    # Create subsubmatrix
                    subsub = []
                    for r in range(1, size):
                        subrow = []
                        for c in range(size):
                            if c != col:
                                subrow.append(mat[r][c])
                        subsub.append(subrow)
                    sign = 1 if col % 2 == 0 else -1
                    det += sign * mat[0][col] * calculate_determinant(subsub)
                return det
            minor_ij = calculate_determinant(submatrix)
            minor_row.append(minor_ij)
        minor_matrix.append(minor_row)

    return minor_matrix
