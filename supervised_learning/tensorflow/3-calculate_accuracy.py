#!/usr/bin/env python3
"""Calculate accuracy of predictions."""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """Calculate the accuracy of a prediction.
    
    Args:
        y (tensor): Placeholder for labels.
        y_pred (tensor): Network's predictions.
    
    Returns:
        tensor: Decimal accuracy of the prediction.
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
