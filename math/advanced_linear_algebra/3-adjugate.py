#!/usr/bin/env python3
"""
Calculates the adjugate matrix of a matrix
"""


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    """
    # Type check
    if not isinstance(matrix, list) or \
       not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Square & non-empty check
    if matrix == [] or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    cof = cofactor(matrix)

    # Transpose cofactor matrix
    adj = [[cof[j][i] for j in range(n)] for i in range(n)]

    return adj


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    """
    n = len(matrix)
    cofactors = []

    for i in range(n):
        row = []
        for j in range(n):
            sub = [
                matrix[r][:j] + matrix[r][j + 1:]
                for r in range(n) if r != i
            ]
            row.append(((-1) ** (i + j)) * determinant(sub))
        cofactors.append(row)

    return cofactors


def determinant(matrix):
    """
    Calculates the determinant of a matrix
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
