#!/usr/bin/env python3
"""
Module that provides a function to perform matrix multiplication on NumPy arrays.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Performs matrix multiplication of two NumPy arrays.

    Args:
        mat1 (numpy.ndarray): First matrix.
        mat2 (numpy.ndarray): Second matrix.

    Returns:
        numpy.ndarray: Result of mat1 multiplied by mat2.
    """
    return np.matmul(mat1, mat2)
