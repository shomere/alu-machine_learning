#!/usr/bin/env python3
"""
Module that provides a function to transpose a matrix (list of lists)
without loops or imports.
"""


def np_transpose(matrix):
    """
    Returns the transpose of a 1D or 2D matrix.

    Args:
        matrix (list): Input 1D or 2D list.

    Returns:
        list: Transposed list.
    """
    if not matrix or not isinstance(matrix[0], list):
        return matrix
    return list(map(list, zip(*matrix)))
