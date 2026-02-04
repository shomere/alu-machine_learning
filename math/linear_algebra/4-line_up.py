#!/usr/bin/env python3
"""
Module that provides a function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): First array of integers or floats.
        arr2 (list): Second array of integers or floats.

    Returns:
        list: A new list containing the element-wise sum of arr1 and arr2,
        or None if the arrays do not have the same shape.
    """
    if len(arr1) != len(arr2):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
