#!/usr/bin/env python3
"""
Module that provides a function to concatenate two matrices along a specific axis.
"""

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along the given axis.

    Args:
        mat1 (numpy.ndarray): First matrix.
        mat2 (numpy.ndarray): Second matrix.
        axis (int): Axis along which to concatenate (default=0).

    Returns:
        numpy.ndarray: Concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
