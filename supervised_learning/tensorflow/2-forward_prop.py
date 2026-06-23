#!/usr/bin/env python3
"""Forward propagation for the neural network."""

import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Create the forward propagation graph.
    
    Args:
        x (tensor): Placeholder for input data.
        layer_sizes (list): Number of nodes in each layer.
        activations (list): Activation functions for each layer.
    
    Returns:
        tensor: Prediction of the network.
    """
    prev = x
    for i in range(len(layer_sizes)):
        prev = create_layer(prev, layer_sizes[i], activations[i])
    return prev
