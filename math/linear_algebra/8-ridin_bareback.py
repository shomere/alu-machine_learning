#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication.
"""


def mat_mul(mat1, mat2):
    """
    Performs matrix multiplication between two 2D matrices.

    Args:
        mat1 (list of lists): First matrix.
        mat2 (list of lists): Second matrix.

    Returns:
        list of lists or None: Resulting matrix or None if multiplication
        is not possible.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat2)):
                total += mat1[i][k] * mat2[k][j]
            row.append(total)
        result.append(row)

    return result
