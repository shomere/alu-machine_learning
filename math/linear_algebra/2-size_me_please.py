#!/usr/bin/env python3
"""
Module that provides a function to calculate the shape of a matrix.
"""


def matrix_shape(matrix):
    """
    Calculates the shape of a matrix.

    Args:
        matrix (list): A multi-dimensional list representing a matrix.

    Returns:
        list: A list of integers representing the shape of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
