#!/usr/bin/env python3
"""Variational Autoencoder implementation."""

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def sampling(args):
    """Sample from the latent space using reparameterization trick.
    
    Args:
        args (tuple): (mean, log_var) tensors.
    
    Returns:
        tensor: Sampled latent vector.
    """
    mean, log_var = args
    epsilon = K.random_normal(shape=K.shape(mean), mean=0.0, stddev=1.0)
    return mean + K.exp(log_var / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.
    
    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the encoder.
        latent_dims (int): Dimensions of the latent space representation.
    
    Returns:
        tuple: (encoder, decoder, auto) models.
    """
    # Encoder
    encoder_input = Input(shape=(input_dims,))
    encoded = encoder_input
    
    for nodes in hidden_layers:
        encoded = Dense(nodes, activation='relu')(encoded)
    
    # Mean and log variance layers
    mean = Dense(latent_dims, activation=None)(encoded)
    log_var = Dense(latent_dims, activation=None)(encoded)
    
    # Sampling layer using reparameterization trick
    z = Lambda(sampling, output_shape=(latent_dims,))([mean, log_var])
    
    encoder = Model(encoder_input, [z, mean, log_var], name='encoder')
    
    # Decoder
    decoder_input = Input(shape=(latent_dims,))
    decoded = decoder_input
    
    for nodes in reversed(hidden_layers):
        decoded = Dense(nodes, activation='relu')(decoded)
    
    decoded = Dense(input_dims, activation='sigmoid')(decoded)
    decoder = Model(decoder_input, decoded, name='decoder')
    
    # Autoencoder with custom loss
    auto_input = Input(shape=(input_dims,))
    z, mean_output, log_var_output = encoder(auto_input)
    decoded_output = decoder(z)
    auto = Model(auto_input, decoded_output, name='autoencoder')
    
    # Custom VAE loss: reconstruction loss + KL divergence
    def vae_loss(y_true, y_pred):
        # Reconstruction loss (binary cross-entropy)
        reconstruction_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        reconstruction_loss = K.sum(reconstruction_loss, axis=-1)
        
        # KL divergence loss
        kl_loss = 1 + log_var_output - K.square(mean_output) - K.exp(log_var_output)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        return K.mean(reconstruction_loss + kl_loss)
    
    auto.compile(optimizer='adam', loss=vae_loss)
    
    return encoder, decoder, auto
