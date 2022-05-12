import numpy as np
import Data
import tensorflow as tf



###########################################
Encoder_Input, Decoder_Input, Decoder_Output= Data.PrepareData("tur.txt")
EngWord = Data.e
TurWord = Data.t
MaxTurLen = Data.MaxLenTur
MaxEngLen = Data.MaxLenEng

Encoder_Input = Encoder_Input[:45000]
Decoder_Input = Decoder_Input[:45000]
Decoder_Output = Decoder_Output[:45000]
###############################################


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, UnitSize):
        super(BahdanauAttention, self).__init__()
        
        self.dense_1 = tf.keras.layers.Dense(UnitSize, use_bias = False)
        self.dense_2 = tf.keras.layers.Dense(UnitSize, use_bias = False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):

        W1_query = self.dense_1(query)
        W2_key= self.dense_2(value)

        query_mask = tf.ones(query.shape[:-1], dtype = bool)
        value_mask = mask
        context_vector, attention_weights = self.attention(
            inputs = [W1_query, value, W2_key],
            mask = [query_mask, value_mask]
            return_attention_scores = True
        ) 
        return context_vector, attention_weights






if __name__ == "__main__":
    Attention = BahdanauAttention(4)
    example  = tf.ones((64, 18))
    
