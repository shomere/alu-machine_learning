#!/usr/bin/env python3
"""Train and save a neural network classifier."""

import tensorflow as tf

calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Build, train, and save a neural network classifier.
    
    Args:
        X_train (numpy.ndarray): Training input data.
        Y_train (numpy.ndarray): Training labels.
        X_valid (numpy.ndarray): Validation input data.
        Y_valid (numpy.ndarray): Validation labels.
        layer_sizes (list): Number of nodes in each layer.
        activations (list): Activation functions for each layer.
        alpha (float): Learning rate.
        iterations (int): Number of iterations to train.
        save_path (str): Path to save the model.
    
    Returns:
        str: Path where the model was saved.
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    
    # Create placeholders
    x, y = create_placeholders(nx, classes)
    
    # Build forward propagation
    y_pred = forward_prop(x, layer_sizes, activations)
    
    # Calculate loss and accuracy
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    
    # Create training operation
    train_op = create_train_op(loss, alpha)
    
    # Add to graph collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    
    # Initialize variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for i in range(iterations + 1):
            # Calculate cost and accuracy for training
            feed_train = {x: X_train, y: Y_train}
            train_cost, train_acc = sess.run([loss, accuracy], feed_dict=feed_train)
            
            # Calculate cost and accuracy for validation
            feed_valid = {x: X_valid, y: Y_valid}
            valid_cost, valid_acc = sess.run([loss, accuracy], feed_dict=feed_valid)
            
            # Print at specified intervals
            if i % 100 == 0 or i == iterations:
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")
            
            # Run training operation (skip on last iteration)
            if i < iterations:
                sess.run(train_op, feed_dict=feed_train)
        
        # Save the model
        save_path = saver.save(sess, save_path)
    
    return save_path
