#!/usr/bin/env python3
"""Calculate softmax cross-entropy loss."""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """Calculate the softmax cross-entropy loss.
    
    Args:
        y (tensor): Placeholder for labels.
        y_pred (tensor): Network's predictions.
    
    Returns:
        tensor: Loss of the prediction.
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss
