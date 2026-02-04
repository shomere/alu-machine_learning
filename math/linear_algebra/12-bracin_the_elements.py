#!/usr/bin/env python3
"""
Module that provides a function to perform element-wise addition, subtraction,
multiplication, and division on matrix-like nested lists or numbers.
"""


def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    on mat1 and mat2 (nested lists or numbers).

    Args:
        mat1, mat2: Nested lists or numbers

    Returns:
        tuple: (sum, difference, product, quotient)
    """

    try:
        # If mat1 is a number, do arithmetic directly
        return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
    except TypeError:
        # Recursively apply element-wise operations for nested lists
        add = list(map(lambda x, y: np_elementwise(x, y)[0], mat1, mat2))
        sub = list(map(lambda x, y: np_elementwise(x, y)[1], mat1, mat2))
        mul = list(map(lambda x, y: np_elementwise(x, y)[2], mat1, mat2))
        div = list(map(lambda x, y: np_elementwise(x, y)[3], mat1, mat2))
        return add, sub, mul, div
