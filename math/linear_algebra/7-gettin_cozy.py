#!/usr/bin/env python3
"""
Module that provides a function to concatenate two 2D matrices
along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specific axis.

    Args:
        mat1 (list of lists): First 2D matrix.
        mat2 (list of lists): Second 2D matrix.
        axis (int): Axis along which to concatenate (0 or 1).

    Returns:
        list of lists or None: New concatenated matrix or None if impossible.
    """
    if axis == 0:
        # same number of columns required
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    if axis == 1:
        # same number of rows required
        if len(mat1) != len(mat2):
            return None
        return [mat1[i][:] + mat2[i][:] for i in range(len(mat1))]

    return None
