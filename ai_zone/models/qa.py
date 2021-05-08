import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Concatenate,  Bidirectional
from question_answering.models import TruncateLayer

def QuestionAnswerabilityModel(hp):
    # Inputs
    passage_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="p_input")
    question_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="q_input")
    
    # Passage processing
    p_vectorize = hp.get("vectorizer")(passage_input)
    p_embed = Embedding(hp.get("vocab_size"), hp.get("embedding_dim"))(p_vectorize)
    
    # Passage stacked LSTM
    p_lstm_1 = LSTM(64, return_sequences=True)(p_embed)
    p_lstm_2 = LSTM(64, name="p_lstm_2", return_sequences=False)(p_lstm_1)
    p_fclayer = Dense(64, name="p_dense", activation="relu")(p_lstm_2)
    
    # Question processing
    q_vectorize = hp.get("vectorizer")(question_input)
    q_embed = hp.get("embedding")(q_vectorize)
    q_truncate = TruncateLayer(hp.get("max_question_len"))(q_embed)
    
    # Question stacked LSTM
    q_lstm_1 = LSTM(64, name="q_lstm_1", return_sequences=True)(q_truncate)
    q_lstm_2 = LSTM(64, name="q_lstm_2", return_sequences=False)(q_lstm_1)
    
    # A fully connected layer for the concatenated output
    concatenate = Concatenate()([p_lstm_2, q_lstm_2])
    fc_1 = Dense(128, name="fc_1", activation="relu")(concatenate)
    fc_2 = Dense(64, name="fc_2", activation="relu")(fc_1)
    fc_3 = Dense(32, name="fc_3", activation="relu")(fc_2)
    
    # Output
    output = Dense(1, activation="sigmoid", name="output")(fc_3)
    
    model = tf.keras.Model(inputs=[passage_input, question_input], outputs=output)
    model.summary()
    return model

def AttentionedAnswerability(hp):
    # Inputs
    passage_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="p_input")
    question_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="q_input")
    
    # Passage processing
    vectorize = hp.get("vectorizer")
    p_vectorize = vectorize(passage_input)
    embed = Embedding(hp.get("vocab_size"), hp.get("embedding_dim"), mask_zero=True)
    p_embed = embed(p_vectorize)
    
    # Passage stacked Bidirectional LSTM
    p_lstm_1 = Bidirectional(LSTM(100, name="p_lstm_1", return_sequences=True))(p_embed)
    p_lstm_2 = LSTM(100, name="p_lstm_2", return_sequences=True)(p_lstm_1)
    
    # Question processing
    q_vectorize = vectorize(question_input)
    q_embed = embed(q_vectorize)
    # q_truncate = TruncateLayer(hp.get("max_question_len"))(q_embed)
    
    # Passage stacked Bidirectional LSTM
    q_lstm_1 = Bidirectional(LSTM(100, name="q_lstm_1", return_sequences=True))(q_embed)
    q_lstm_2 = LSTM(100, name="q_lstm_2", return_sequences=True)(q_lstm_1)
    
    # Attention everybody, a new layer's coming to town
    attention_layer = tf.keras.layers.Attention(name="attention")
    attn_out = attention_layer([p_lstm_2, q_lstm_2])
    question_encoding = tf.keras.layers.Concatenate()([q_lstm_2, attn_out])
    
    # A fully connected layer for the concatenated output
    last_lstm = Bidirectional(LSTM(100, name="q_lstm_1", return_sequences=False))(question_encoding)
    fc_1 = Dense(128, name="fc_1", activation="relu")(last_lstm)
    fc_2 = Dense(64, name="fc_2", activation="relu")(fc_1)
    fc_3 = Dense(32, name="fc_3", activation="relu")(fc_2)
    
    # Output
    output = Dense(1, activation="sigmoid", name="output")(fc_3)
    
    model = tf.keras.Model(inputs=[passage_input, question_input], outputs=output)
    model.summary()
    return model


def AttentionedAnswerIndex(hp):
    # Inputs
    passage_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="p_input")
    question_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="q_input")
    
    # Passage processing
    vectorize = hp.get("vectorizer")
    p_vectorize = vectorize(passage_input)
    embed = Embedding(hp.get("vocab_size"), hp.get("embedding_dim"), mask_zero=True)
    p_embed = embed(p_vectorize)
    
    # Passage stacked Bidirectional LSTM
    p_lstm_1 = Bidirectional(LSTM(100, name="p_lstm_1", return_sequences=True))(p_embed)
    p_lstm_2 = LSTM(100, name="p_lstm_2", return_sequences=True)(p_lstm_1)
    
    # Question processing
    q_vectorize = vectorize(question_input)
    q_embed = embed(q_vectorize)
    # q_truncate = TruncateLayer(hp.get("max_question_len"))(q_embed)
    
    # Passage stacked Bidirectional LSTM
    q_lstm_1 = Bidirectional(LSTM(100, name="q_lstm_1", return_sequences=True))(q_embed)
    q_lstm_2 = LSTM(100, name="q_lstm_2", return_sequences=True)(q_lstm_1)
    
    # Attention everybody, a new layer's coming to town
    attention_layer = tf.keras.layers.Attention(name="attention")
    attn_out = attention_layer([p_lstm_2, q_lstm_2])
    question_encoding = tf.keras.layers.Concatenate()([q_lstm_2, attn_out])
    
    # A fully connected layer for the concatenated output
    last_lstm = Bidirectional(LSTM(100, name="q_lstm_1", return_sequences=False))(question_encoding)
    fc_1 = Dense(128, name="fc_1", activation="relu")(last_lstm)
    fc_2 = Dense(64, name="fc_2", activation="relu")(fc_1)
    fc_3 = Dense(32, name="fc_3", activation="relu")(fc_2)
    
    # Output
    output = Dense(2, name="indexes")(fc_3)
    
    model = tf.keras.Model(inputs=[passage_input, question_input], outputs=output)
    model.summary()
    return model


def QAFirstIndex(hp):
    # Inputs
    passage_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="p_input")
    question_input = tf.keras.Input(shape=(1,), dtype=tf.string, name="q_input")
    
    # Passage processing
    p_vectorize = hp.get("vectorizer")(passage_input)
    p_embed = Embedding(hp.get("vocab_size"), hp.get("embedding_dim"))(p_vectorize)
    
    # Passage stacked LSTM
    p_lstm_1 = LSTM(64, name="p_lstm_1", return_sequences=True)(p_embed)
    p_lstm_2 = LSTM(64, name="p_lstm_2", return_sequences=False)(p_lstm_1)
    
    # Question processing
    q_vectorize = hp.get("vectorizer")(question_input)
    q_embed = hp.get("embedding")(q_vectorize)
    q_truncate = TruncateLayer(hp.get("max_question_len"))(q_embed)
    
    # Question stacked LSTM
    q_lstm_1 = LSTM(64, name="q_lstm_1", return_sequences=True)(q_truncate)
    q_lstm_2 = LSTM(64, name="q_lstm_2", return_sequences=False)(q_lstm_1)
    
    # A fully connected layer for the concatenated output
    concatenate = Concatenate(axis=-1)([p_lstm_2, q_lstm_2])
    fc_1 = Dense(128, name="fc_1", activation="relu")(concatenate)
    fc_2 = Dense(64, name="fc_2", activation="relu")(fc_1)
    fc_3 = Dense(32, name="fc_3", activation="relu")(fc_2)
    
    # Output
    output = Dense(1, name="o_start_index")(fc_3)
    
    model = tf.keras.Model(inputs=[passage_input, question_input], outputs=output)
    model.summary()
    return model

def Seq2Seq(hp):
    # Variables
    latent_dim = 1024
    
    # Inputs
    encoder_inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="e_input")
    
    # Passage processing
    vectorize = hp.get("vectorizer")
    embed = Embedding(hp.get("vocab_size") + 1, hp.get("embedding_dim"), mask_zero=True)
    encoder_vectorize = vectorize(encoder_inputs)
    encoder_embed = embed(encoder_vectorize)
    
    # Encoder - keep the states
    encoder = LSTM(latent_dim, name="encoder_lstm", return_state=True)
    _, state_h, state_c = encoder(encoder_embed)
    encoder_states = [state_h, state_c]
    
    # Set up the decoder with the encoder states as initial
    decoder_inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="d_input")
    
    # Vectorize and embed
    decoder_vectorize = vectorize(decoder_inputs)
    decoder_embed = embed(decoder_vectorize)
    truncate = TruncateLayer(hp.get("max_answer_len"))
    decoder_truncate = truncate(decoder_embed)
    
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the 
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, name="lstm", return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_truncate,
                                        initial_state=encoder_states)
    decoder_dense = Dense(hp.get("vocab_size"), activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Initialize and return the model
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    
    # INFERENCE
    # Encoder submodel needed for inference
    encoder_inf_model = tf.keras.Model(encoder_inputs, encoder_states)
    
    # DECODER SUBMODEL
    # Decoder submodel inputs
    decoder_state_input_h = tf.keras.Input(shape=(latent_dim,))
    decoder_state_input_c = tf.keras.Input(shape=(latent_dim,))
    decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Decoder submodel definition and steps to output
    decoder_outputs, state_h, state_c = decoder_lstm(truncate(embed(vectorize(decoder_inputs))), initial_state=decoder_state_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_inf_model = tf.keras.Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)
    
    # Return all models
    return model, encoder_inf_model, decoder_inf_model