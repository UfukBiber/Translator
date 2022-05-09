import numpy as np
import Data_V2
import tensorflow as tf

###########################################
Encoder_Input, Dec_For_Input, Dec_For_Output, Dec_Back_Inp, Dec_Back_Out = Data_V2.PrepareData("tur.txt")
EngWord = Data_V2.e
TurWord = Data_V2.t
MaxTurLen = Data_V2.MaxLenTur
MaxEngLen = Data_V2.MaxLenEng

############################################



class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.80
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > self.acc:
            print("\n\nReached %f percent accuracy"%self.acc)
            print("Model is saved")
            self.acc += 0.05
            self.model.save("my_model_tur")
        if logs.get("accuracy")>0.98:
            self.model.stop_training = True
            

callback = MyCallBack()

##########################ENCODER###########################################
input_encoder = tf.keras.layers.Input(shape = (Encoder_Input.shape[-1]))
embedding_1 = tf.keras.layers.Embedding(EngWord+1, 256, name = "Embedding_1")(input_encoder)
_, forward_state_h, forward_state_c, backward_state_h, backward_state_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_state = True, name = "LSTM_1"))(embedding_1)
encoder_for_state = [forward_state_h, forward_state_c]
encoder_back_state = [backward_state_h, backward_state_c]



#############################################################################################
EmbeddingLayer = tf.keras.layers.Embedding(EngWord+1, 256, name = "Embedding_2")

######################## DECODER Forward Layer ##############################################

input_for_decoder = tf.keras.layers.Input(shape = (Dec_For_Input.shape[-1]), name = "input_for_decoder")
output_forward = EmbeddingLayer(input_for_decoder)
output_forward, _,  _ = tf.keras.layers.LSTM(256, return_sequences = True, return_state = True, name = "LSTM_2")(output_forward, initial_state = encoder_for_state)
#############################################################################

######################## Decoder Backward Layer #############################################
input_back_decoder = tf.keras.layers.Input(shape = (Dec_Back_Inp.shape[-1]), name = "input_back_decoder")
output_backward = EmbeddingLayer(input_back_decoder) 
output_backward, _,  _ = tf.keras.layers.LSTM(256, return_sequences = True, return_state = True, name = "LSTM_3")(output_backward, initial_state = encoder_back_state)

#############################################################################################

output = tf.keras.layers.Concatenate()([output_forward, output_backward])

output = tf.keras.layers.Dense(TurWord+1, activation = "softmax")(output)

model = tf.keras.models.Model([input_encoder, input_for_decoder, input_back_decoder], output)





model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


if __name__ == "__main__":
    model.fit([Encoder_Input, Dec_For_Input, Dec_Back_Inp], Dec_For_Output , epochs = 50, callbacks = [callback], validation_split = 0.1)

