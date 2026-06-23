#!/usr/bin/env python3
"""Create placeholders for the neural network."""

import tensorflow as tf


def create_placeholders(nx, classes):
    """Create placeholders for input data and labels.
    
    Args:
        nx (int): Number of feature columns in the data.
        classes (int): Number of classes in the classifier.
    
    Returns:
        tuple: (x, y) placeholders for input data and one-hot labels.
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
