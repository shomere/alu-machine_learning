#!/usr/bin/env python3
"""
Module that provides a function to transpose a NumPy array.
"""

import numpy as np


def np_transpose(matrix):
    """
    Transposes the given NumPy array.

    Args:
        matrix (numpy.ndarray): Input array to transpose.

    Returns:
        numpy.ndarray: Transposed array.
    """
    return np.transpose(matrix)
