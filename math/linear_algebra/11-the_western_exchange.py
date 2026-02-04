#!/usr/bin/env python3
"""
Matrix Transposition Module

This module provides a function to transpose matrices using NumPy.
The function handles matrices of various dimensions appropriately.
"""


def np_transpose(matrix):
    """
    Transposes a matrix.

    Args:
        matrix: A numpy.ndarray to be transposed

    Returns:
        numpy.ndarray: The transposed matrix
    """
    return matrix.transpose()
