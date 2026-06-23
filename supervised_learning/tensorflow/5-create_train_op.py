#!/usr/bin/env python3
"""Create training operation."""

import tensorflow as tf


def create_train_op(loss, alpha):
    """Create the training operation using gradient descent.
    
    Args:
        loss (tensor): Loss of the network's prediction.
        alpha (float): Learning rate.
    
    Returns:
        operation: Training operation.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    train_op = optimizer.minimize(loss)
    return train_op
