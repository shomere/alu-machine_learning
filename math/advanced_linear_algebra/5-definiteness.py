#!/usr/bin/env python3
"""
Matrix Definiteness Calculation Module

This module provides a function to determine the definiteness
of a symmetric matrix using eigenvalue analysis.
"""

import numpy as np


def definiteness(matrix):
    """
    Determine the definiteness of a symmetric matrix.

    Args:
        matrix: A numpy.ndarray of shape (n, n)

    Returns:
        str: The definiteness classification or None if invalid

    Raises:
        TypeError: If matrix is not a numpy.ndarray
    """
    # Check if matrix is a numpy array
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is 2D and square
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    # Check if matrix is symmetric (within tolerance)
    if not np.allclose(matrix, matrix.T):
        return None

    # Calculate eigenvalues
    try:
        eigenvalues = np.linalg.eigvals(matrix)
    except np.linalg.LinAlgError:
        return None

    # Check if eigenvalues are all real (should be for symmetric matrix)
    if not np.all(np.isreal(eigenvalues)):
        return None

    # Convert to real numbers (imaginary part should be negligible)
    eigenvalues = np.real(eigenvalues)

    # Classify based on eigenvalues
    all_positive = np.all(eigenvalues > 0)
    all_non_negative = np.all(eigenvalues >= 0)
    all_negative = np.all(eigenvalues < 0)
    all_non_positive = np.all(eigenvalues <= 0)

    # Check for exact zero with tolerance
    has_zero = np.any(np.abs(eigenvalues) < 1e-10)

    if all_positive:
        return "Positive definite"
    elif all_non_negative and has_zero:
        return "Positive semi-definite"
    elif all_negative:
        return "Negative definite"
    elif all_non_positive and has_zero:
        return "Negative semi-definite"
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "Indefinite"
    else:
        return None
