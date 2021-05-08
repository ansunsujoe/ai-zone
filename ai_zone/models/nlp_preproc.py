import tensorflow as tf
from tensorflow.keras.layers import Embedding

def TextVectorizer(vectorizer_layer) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string),
        vectorizer_layer
    ])

def EmbeddingLayer(embedding_matrix, input_length):
    return Embedding(
        embedding_matrix.shape[0],
        embedding_matrix.shape[1],
        input_length=input_length,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
        mask_zero=True
    )
    
def TextEmbedder(vectorizer_layer, embedding_layer) -> tf.keras.Model:
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,), dtype=tf.string),
        vectorizer_layer,
        embedding_layer
    ])

class TruncateLayer(tf.keras.layers.Layer):
    def __init__(self, new_length):
        super(TruncateLayer, self).__init__()
        self.new_length = new_length

    def call(self, inputs):
        return inputs[:, 0:self.new_length, :]
    
    def compute_output_shape(self, input_shape):
        print(input_shape)
        return (None, self.new_length)
    
    def get_config(self):
        return {"question_size": self.new_length}