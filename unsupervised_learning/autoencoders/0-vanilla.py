#!/usr/bin/env python3
"""Vanilla Autoencoder implementation."""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a vanilla autoencoder.
    
    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.
    
    Returns:
        tuple: (encoder, decoder, auto) models.
    """
    # Encoder
    encoder_input = keras.layers.Input(shape=(input_dims,))
    encoded = encoder_input
    
    for nodes in hidden_layers:
        encoded = keras.layers.Dense(nodes, activation='relu')(encoded)
    
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)
    encoder = keras.models.Model(encoder_input, encoded, name='encoder')
    
    # Decoder
    decoder_input = keras.layers.Input(shape=(latent_dims,))
    decoded = decoder_input
    
    for nodes in reversed(hidden_layers):
        decoded = keras.layers.Dense(nodes, activation='relu')(decoded)
    
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.models.Model(decoder_input, decoded, name='decoder')
    
    # Autoencoder
    auto_input = keras.layers.Input(shape=(input_dims,))
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = keras.models.Model(auto_input, decoded_output, name='autoencoder')
    
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
