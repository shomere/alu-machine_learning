#!/usr/bin/env python3
"""Sparse Autoencoder implementation with L1 regularization."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, ActivityRegularization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """Create a sparse autoencoder with L1 regularization.
    
    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.
        lambtha (float): Regularization parameter for L1 regularization.
    
    Returns:
        tuple: (encoder, decoder, auto) models.
    """
    # Encoder
    encoder_input = Input(shape=(input_dims,))
    encoded = encoder_input
    
    for nodes in hidden_layers:
        encoded = Dense(nodes, activation='relu')(encoded)
    
    # Add L1 regularization on the encoded output
    encoded = Dense(latent_dims, activation='relu', activity_regularizer=l1(lambtha))(encoded)
    encoder = Model(encoder_input, encoded, name='encoder')
    
    # Decoder
    decoder_input = Input(shape=(latent_dims,))
    decoded = decoder_input
    
    for nodes in reversed(hidden_layers):
        decoded = Dense(nodes, activation='relu')(decoded)
    
    decoded = Dense(input_dims, activation='sigmoid')(decoded)
    decoder = Model(decoder_input, decoded, name='decoder')
    
    # Autoencoder
    auto_input = Input(shape=(input_dims,))
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = Model(auto_input, decoded_output, name='autoencoder')
    
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
