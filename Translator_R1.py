import os
import numpy as np 
import re
# import tensorflow as tf


def generate_data(data):
    eng_words, tur_words = {}, {}
    vect_eng, vect_tur = [], []
    max_english_seq, max_tur_seq = 0, 0
    max_english_words, max_tur_words =0, 0
    with open(data, 'r', encoding = "utf-8") as f:
        lines = f.read().lower().split("\n")
        for line in lines:
            line = line.split("\t")
            

        

def generate_model(max_inp_words,embedding_dim, inp, max_tar_words, lstm_dim, tar ):
    encoder_input = tf.keras.Input(shape = (None, max_inp_words))
    vector = tf.keras.layers.Embedding(max_inp_words, embedding_dim)(encoder_input)
    output, state_h, state_c = tf.keras.layers.LSTM(lstm_dim, return_state = True)(vector)
    decoder_input = tf.keras.Input(shape = None)
    state = [state_h, state_c]
    vector_1 = tf.keras.layers.Embedding(max_tar_words, embedding_dim)(decoder_input)
    output= tf.keras.layers.LSTM(lstm_dim, return_sequences = True)(vector_1, initial_state = state)
    dense = tf.keras.layers.Dense(max_tar_words, activation = "softmax")(output)
    model = tf.keras.Model([encoder_input, decoder_input], dense)
    return model





        
if __name__ == "__main__" :
    generate_data("C:\\Users\\ufuk\\Desktop\\Translator\\tur.txt")

    
    
    

