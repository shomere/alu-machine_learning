#!/usr/bin/env python3
"""
Element-wise Operations Module

This module provides a function for performing element-wise
arithmetic operations on numpy arrays.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication,
    and division on two numpy arrays.

    Args:
        mat1: First numpy.ndarray
        mat2: Second numpy.ndarray or scalar value

    Returns:
        tuple: A tuple containing (sum, difference, product, quotient)
               in that order
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
