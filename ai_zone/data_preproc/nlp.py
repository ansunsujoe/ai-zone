from pathlib import Path
from typing import List
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from question_answering import models
import logging
from string import punctuation

logger = logging.getLogger('data_preproc')


def preprocess(text:str):
    word_array = text.split()
    for i in range(len(word_array)):
        word_array[i] = word_array[i].strip(punctuation).lower()
    return " ".join(word_array)
    
    # sentences = nltk.sent_tokenize(text)
    # for sentence in sentences:   
    #     word_array += nltk.word_tokenize(sentence)
    # word_array = [word for word in word_array if len(word) > 2 or word.isalnum()]
    # return " ".join(word_array)

def load_embeddings_file(path:Path):
    # Load glove file
    embeddings_index = {}
    with open(path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    return embeddings_index

def get_subarray_indices(subarray, array) -> List[int]:
    """
    Get the start index and length of a subarray or subsequence
    
    Args:
        subarray ([type]): Subsequence
        array ([type]): Original sequence

    Returns:
        List: First argument is the start index and second is the length
        of the sequence
    """
    window_size = len(subarray)
    for i in range(len(array) - window_size + 1):
        if subarray == array[i:i+window_size]:
            return [i, window_size]
    return [-1, -1]

def initialize_vectorizer_layer(text, pad_length, max_tokens=None):
    # Create vectorizer
    vectorizer = TextVectorization(output_sequence_length=pad_length, standardize=None, max_tokens=max_tokens)
    vectorizer.adapt(text)
    vocab = vectorizer.get_vocabulary()
    return vectorizer, vocab

def initialize_glove_layer(path:Path, vocab, input_length, embedding_dim):
    # Load glove file
    embeddings_index = load_embeddings_file(path)
    print("Found %s word vectors." % len(embeddings_index))
    
    word_index = dict(zip(vocab, range(len(vocab))))
    
    # Variable initializations
    num_tokens = len(vocab) + 2
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
            
    # Return final values
    print("Converted %d words (%d misses)" % (hits, misses))
    embedding_layer = models.EmbeddingLayer(embedding_matrix, input_length=input_length)
    return embedding_layer

def onehot_vectorize(text_array, vectorizer, vocab_length):
    one_hot_array = []
    for i in range(0, len(text_array), 64):
        vectorized_array = vectorizer(text_array[i:i+64])
        one_hot_array.append(tf.keras.utils.to_categorical(vectorized_array, num_classes=vocab_length))
    return np.concatenate(one_hot_array, axis=0)