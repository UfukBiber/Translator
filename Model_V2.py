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

############################################



class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.80
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > self.acc:
            print("\n\nReached %f percent accuracy"%self.acc)
            print("Model is saved")
            self.acc += 0.05
            self.model.save("my_model_tur_V2")
        if logs.get("accuracy")>0.98:
            self.model.stop_training = True
            self.model.save("my_model_tur_V2")
            

callback = MyCallBack()

##########################ENCODER###########################################
input_encoder = tf.keras.layers.Input(shape = (Encoder_Input.shape[-1]), name = "input_encoder")
embedding_1 = tf.keras.layers.Embedding(EngWord+1, 256, name = "Embedding_1")(input_encoder)
_, forward_state_h, forward_state_c, backward_state_h, backward_state_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_state = True, name = "LSTM_1"))(embedding_1)
encoder_state_h = tf.keras.activations.softmax()(tf.math.add(forward_state_h, backward_state_h))
encoder_state_c = tf.keras.activations.softmax()(tf.math.add(forward_state_c, backward_state_c))
encoderStates = [encoder_state_h, encoder_state_c]


#############################################################################################

# ######################## DECODER  ##############################################

input_decoder = tf.keras.layers.Input(shape = (Decoder_Input.shape[-1]), name = "input_decoder")
output = tf.keras.layers.Embedding(TurWord+1, 256, name = "Embedding_2")(input_decoder)
output, _,  _ = tf.keras.layers.LSTM(256, return_sequences = True, return_state = True, name = "LSTM_2")(output, initial_state = encoderStates)
output = tf.keras.layers.Dense(TurWord+1, activation = "softmax")(output)
#############################################################################



model = tf.keras.models.Model([input_encoder, input_decoder], output)




model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


if __name__ == "__main__":
    model.fit([Encoder_Input, Decoder_Input], Decoder_Output , epochs = 50, callbacks = [callback], validation_split = 0.1)

