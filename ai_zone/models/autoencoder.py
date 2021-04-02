import tensorflow as tf
from typing import Tuple
from tensorflow.keras.layers import *

def LSTMAutoencoder(
    timesteps:int,
    features:int,
    latent_space:int
) -> Tuple(tf.keras.Model, tf.keras.Model, tf.keras.Model):
    """
    Autoencoder that compresses Time-Series data to an n-dimensional vector latent space.
    Sam Black, Using LSTM Autoencoders on multidimensional time-series data, Towards Data Science, https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1

    Args:
        timesteps: number of timesteps
        features: features in each timestep
        latent_space: number of dimensions in the encoded vector
    """
    # Create the encoder
    encoder_input = tf.keras.Input(shape=(timesteps, features), name="encoder_input")
    encoder_1 = LSTM(64, name="encoder_1", kernel_initializer="he_uniform", return_sequences=True)(encoder_input)
    encoder_2 = LSTM(32, name="encoder_2", kernel_initializer="he_uniform", return_sequences=True)(encoder_1)
    encoder_output = LSTM(latent_space, name="encoder_3", kernel_initializer="he_uniform", return_sequences=False)(encoder_2)

    encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")
    encoder.summary()

    # Create the decoder
    decoder_input = tf.keras.Input(shape=(latent_space,))
    encoder_decoder_bridge = RepeatVector(timesteps, name="encoder_decoder_bridge")(decoder_input)
    decoder_1 = LSTM(latent_space, name="decoder_1", kernel_initializer="he_uniform", return_sequences=True)(encoder_decoder_bridge)
    decoder_2 = LSTM(32, name="decoder_2", kernel_initializer="he_uniform", return_sequences=True)(decoder_1)
    decoder_3 = LSTM(64, name="decoder_3", kernel_initializer="he_uniform", return_sequences=True)(decoder_2)
    decoder_output = TimeDistributed(Dense(features))(decoder_3)

    decoder = tf.keras.Model(decoder_input, decoder_output, name="decoder")
    decoder.summary()

    # Create the autoencoder
    autoencoder_input = tf.keras.Input(shape=(timesteps, features))
    encoded_data = encoder(autoencoder_input)
    decoded_data = decoder(encoded_data)

    autoencoder = tf.keras.Model(autoencoder_input, decoded_data, name="autoencoder")
    autoencoder.summary()

    return autoencoder, encoder, decoder