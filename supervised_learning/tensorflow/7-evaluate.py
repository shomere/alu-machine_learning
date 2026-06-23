#!/usr/bin/env python3
"""Evaluate a neural network."""

import tensorflow as tf


def evaluate(X, Y, save_path):
    """Evaluate the output of a neural network.
    
    Args:
        X (numpy.ndarray): Input data to evaluate.
        Y (numpy.ndarray): One-hot labels for X.
        save_path (str): Location to load the model from.
    
    Returns:
        tuple: (prediction, accuracy, loss)
    """
    with tf.Session() as sess:
        # Import the meta graph
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        
        # Get tensors from the graph's collection
        graph = tf.get_default_graph()
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        loss = tf.get_collection('loss')[0]
        accuracy = tf.get_collection('accuracy')[0]
        
        # Evaluate
        feed_dict = {x: X, y: Y}
        pred, acc, cost = sess.run([y_pred, accuracy, loss], feed_dict=feed_dict)
        
        return pred, acc, cost
