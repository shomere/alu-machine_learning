#!/usr/bin/env python3
"""Create a layer for the neural network."""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Create a layer with He et. al initialization.
    
    Args:
        prev (tensor): Output tensor of the previous layer.
        n (int): Number of nodes in the layer.
        activation: Activation function for the layer.
    
    Returns:
        tensor: Output tensor of the layer.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer"
    )
    return layer(prev)
