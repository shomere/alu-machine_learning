#!/usr/bin/env python3
"""
Module that provides a function to transpose a matrix (nested lists).
"""


def np_transpose(matrix):
    """
    Returns the transpose of a matrix (list of lists).

    Args:
        matrix (list): Input 1D or 2D list.

    Returns:
        list: Transposed list.
    """
    if not matrix or not isinstance(matrix[0], list):
        return matrix  # 1D or empty, return as is
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
