#!/usr/bin/env python3
"""Convolutional Autoencoder implementation."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model


def autoencoder(input_dims, filters, latent_dims):
    """Create a convolutional autoencoder.
    
    Args:
        input_dims (tuple): Dimensions of the model input (height, width, channels).
        filters (list): Number of filters for each convolutional layer in the encoder.
        latent_dims (tuple): Dimensions of the latent space representation.
    
    Returns:
        tuple: (encoder, decoder, auto) models.
    """
    # Encoder
    encoder_input = Input(shape=input_dims)
    encoded = encoder_input
    
    for f in filters:
        encoded = Conv2D(f, (3, 3), activation='relu', padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    
    encoded = Conv2D(latent_dims[2], (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoder = Model(encoder_input, encoded, name='encoder')
    
    # Decoder
    decoder_input = Input(shape=latent_dims)
    decoded = decoder_input
    
    # Upsample to match the last encoder layer
    decoded = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(latent_dims[2], (3, 3), activation='relu', padding='same')(decoded)
    
    # Decoder layers with upsampling
    for f in reversed(filters):
        decoded = UpSampling2D((2, 2))(decoded)
        decoded = Conv2D(f, (3, 3), activation='relu', padding='same')(decoded)
    
    # Second to last convolution with valid padding
    decoded = Conv2D(input_dims[2], (3, 3), activation='sigmoid', padding='valid')(decoded)
    decoder = Model(decoder_input, decoded, name='decoder')
    
    # Autoencoder
    auto_input = Input(shape=input_dims)
    encoded_output = encoder(auto_input)
    decoded_output = decoder(encoded_output)
    auto = Model(auto_input, decoded_output, name='autoencoder')
    
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
