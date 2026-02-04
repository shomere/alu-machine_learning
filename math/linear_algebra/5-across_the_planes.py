#!/usr/bin/env python3
"""
Module that provides a function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of lists): First 2D matrix of ints/floats.
        mat2 (list of lists): Second 2D matrix of ints/floats.

    Returns:
        list of lists: A new matrix containing the element-wise sum of mat1
        and mat2, or None if the matrices do not have the same shape.
    """
    if len(mat1) != len(mat2):
        return None

    new_matrix = []

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None
        new_matrix.append(
            [mat1[i][j] + mat2[i][j] for j in range(len(mat1[i]))]
        )

    return new_matrix
