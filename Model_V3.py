import numpy as np
import Data
import tensorflow as tf

QUANTITY = 50

TRAIN = int(0.9 * 50000)

###########################################
Encoder_Input, Decoder_Input, Decoder_Output= Data.PrepareData("tur.txt", QUANTITY)
EngWord = Data.e
TurWord = Data.t
MaxTurLen = Data.MaxLenTur
MaxEngLen = Data.MaxLenEng

Encoder_Input = Encoder_Input[:TRAIN]
Decoder_Input = Decoder_Input[:TRAIN]
Decoder_Output = Decoder_Output[:TRAIN]
###############################################


class Encoder(tf.keras.layers.Layer):
    def __init__(self, unitsize, embedding_size):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(EngWord+1, embedding_size)
        self.Rnn = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(unitsize, return_state = True, return_sequences = True))

    def call(self, input):
        output = self.embedding(input)
        output, for_state_h, for_state_c, back_state_h, back_state_c = self.Rnn(output)
        state_h = tf.concat([for_state_h, back_state_h], axis = -1)
        state_c = tf.concat([for_state_c, back_state_c], axis = -1)
        return output, state_h, state_c


class Decoder(tf.keras.layers.Layer):
    def __init__(self, unitsize, embedding_size):
        super(Decoder, self).__init__()
        embedding = tf.keras.layers.Embedding(TurWord+1, embedding_size)
        pass





if __name__ == "__main__":
    Attention = Encoder(256, 256)
    y, state_h, state_c = Attention(Encoder_Input)
    print(y.shape)
    print(state_h.shape)
    print(state_c.shape)

    
