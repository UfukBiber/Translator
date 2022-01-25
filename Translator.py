import os
import numpy as np 
import re
import tensorflow as tf
class Data():
    def __init__(self):
        self.encoder_input, self.decoder_input, self.decoder_out, self.word_token_eng, self.word_token_tur = self.vectorize_sentences()
        self.max_eng_token = max(self.word_token_eng.values())
        self.max_tur_token = max(self.word_token_tur.values())
        self.max_eng_seq_len = self.encoder_input.shape[1]
        self.max_tur_seq_len = self.decoder_input.shape[1]
        print(self.max_eng_seq_len, self.max_tur_seq_len)
    def read_data(self):
        eng, tur = [], []
        eng_words, tur_words = set(), set()
        with open("C:\\Users\\ufuk\\Desktop\\Translator\\tur.txt", "r", encoding = "UTF-8") as file:
            lines = file.read().split("\n")
            lines.pop()
            file.close()
        for line in lines[-20000:-15000]:
            line = re.split(r"\t", line)
            sentence = re.sub(r"[!.?;#$/,'\*\-\"]"," ", line[0]).lower().split()
            eng.append(sentence)
            for i in sentence:
                eng_words.add(i)
            sentence = re.sub(r"[!.:;?#$/,'\*\-\"]", " ", line[1]).lower().split()  
            tur.append(sentence)
            for i in sentence:
                tur_words.add(i)
        return eng, tur, eng_words, tur_words
    def tokenize_words(self, eng_words, tur_words):
        word_tokens_eng = {"start" : 1, "end" : 2, "unk" : 3}
        word_tokens_tur = {"start" : 1, "end" : 2, "unk" : 3}
        n = 3
        for i in eng_words :
            if word_tokens_eng.get(i) == None:
                n += 1
                word_tokens_eng[i] = n
        n = 3
        for i in tur_words :
            if word_tokens_tur.get(i) == None:
                n += 1
                word_tokens_tur[i] = n
        return word_tokens_eng, word_tokens_tur
    def vectorize_sentences(self):
        eng, tur, eng_words, tur_words= self.read_data()
        max_eng_seq = 101 #max([len(i) for i in eng])
        max_tur_seq = 71 #max([len(i) for i in tur])
        word_token_eng, word_token_tur = self.tokenize_words(eng_words, tur_words)
        encoder_inp_eng, decoder_inp_tur = np.zeros(shape= (len(eng), max_eng_seq + 2), dtype = np.uint32), np.zeros(shape = (len(tur), max_tur_seq + 2), dtype = np.uint32) 
        decoder_out = np.zeros(shape = (len(tur), max_tur_seq + 2), dtype = np.uint32) 
        encoder_inp_eng[:,0] = 1
        decoder_inp_tur[:,0] = 1 
        for i in range(len(eng)):     
            for j in range(len(eng[i])):     
                encoder_inp_eng[i, j+1] = word_token_eng[eng[i][j]]
            encoder_inp_eng[i, j+2] = 2
            for j in range(len(tur[i])):
                decoder_inp_tur[i, j+1] = word_token_tur[tur[i][j]]
                decoder_out[i, j] = word_token_tur[tur[i][j]]
            decoder_inp_tur[i, j+2] = 2
            decoder_out[i, j+1] = 2                                                                  
        return encoder_inp_eng, decoder_inp_tur, decoder_out, word_token_eng, word_token_tur

def get_model(max_eng_token, max_tur_token, embedding_dim, lstm_units, max_eng_seq, max_tur_seq):
    encoder_input = tf.keras.Input(shape = ( max_eng_seq))
    encoder_embedding_out = tf.keras.layers.Embedding(max_eng_token + 1, embedding_dim)(encoder_input)
    _, state_c, state_h = tf.keras.layers.LSTM(lstm_units, return_state = True)(encoder_embedding_out)
    states = [state_c, state_h]
    decoder_input = tf.keras.Input(shape = ( max_tur_seq))
    decoder_embedding_out = tf.keras.layers.Embedding(max_tur_token + 1, embedding_dim)(decoder_input)
    output = tf.keras.layers.LSTM(lstm_units, return_sequences = True)(decoder_embedding_out, initial_state = states)
    output = tf.keras.layers.Dropout(0.5)(output)
    output = tf.keras.layers.Dense(32, activation = "relu")(output)
    output = tf.keras.layers.Dense(max_tur_token + 1, activation = "softmax")(output)
    model = tf.keras.Model([encoder_input, decoder_input], output)
    return model





        
if __name__ == "__main__" :
    data = Data()
    encoder_inp, decoder_inp = data.encoder_input, data.decoder_input
    output = data.decoder_out
    max_english_token, max_tur_token = data.max_eng_token, data.max_tur_token
    max_eng_seq_len, max_tur_seq_len = data.max_eng_seq_len, data.max_tur_seq_len
    # model = get_model(max_english_token, max_tur_token, 64, 128, max_eng_seq_len, max_tur_seq_len)
    # model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # model.fit([encoder_inp, decoder_inp], output, batch_size = 64, epochs = 3, validation_split = 0.1 )
    # model.save("C:\\Users\\ufuk\\Desktop\\Translator\\my_model.h5")
    model = tf.keras.models.load_model("C:\\Users\\ufuk\\Desktop\\Translator\\my_model.h5")
    x, y = encoder_inp , decoder_inp
    z = output
    model.predict([x[0], y[0]])
    

    
    
    

