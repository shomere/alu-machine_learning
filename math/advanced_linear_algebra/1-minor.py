#!/usr/bin/env python3
"""
Calculates the minor matrix of a matrix
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    """
    # Type check: must be a list of lists
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Must be non-empty square matrix
    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    minors = []
    for i in range(n):
        row_minors = []
        for j in range(n):
            sub = [
                matrix[r][:j] + matrix[r][j + 1:]
                for r in range(n) if r != i
            ]

            det = sub[0][0] * sub[1][1] - sub[0][1] * sub[1][0] \
                if len(sub) == 2 else determinant(sub)

            row_minors.append(det)
        minors.append(row_minors)

    return minors


def determinant(matrix):
    """
    Recursively calculates determinant
    """
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - \
               matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        sub = [
            row[:c] + row[c + 1:]
            for row in matrix[1:]
        ]
        det += ((-1) ** c) * matrix[0][c] * determinant(sub)

    return det
