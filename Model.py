import numpy as np
import Data 
import tensorflow as tf

###########################################
Decoder_Input, Decoder_Output, Encoder_Input = Data.PrepareData("tur.txt")
EngWord = Data.e
TurWord = Data.t
MaxTurLen = Data.MaxLenTur
MaxEngLen = Data.MaxLenEng

############################################

Decoder_Input_train = Decoder_Input
Encoder_Input_train = Encoder_Input
Decoder_Output_train = Decoder_Output

#############################################

print(Decoder_Input_train.shape, Encoder_Input_train.shape, Decoder_Output_train.shape)

class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.80
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > self.acc:
            print("\n\nReached %f percent accuracy"%self.acc)
            print("Model is saved")
            self.acc += 0.05
            self.model.save("my_model_tur")
        if logs.get("accuracy")>0.95:
            self.model.stop_training = True
            

callback = MyCallBack()

##########################ENCODER###########################################
input_encoder = tf.keras.layers.Input(shape = (Encoder_Input_train.shape[-1]))
embedding_1 = tf.keras.layers.Embedding(EngWord+1, 256)(input_encoder)
_, state_h, state_c = tf.keras.layers.LSTM(256, return_state = True)(embedding_1)
encoder_state = [state_h, state_c]


########################DECODER##############################################

input_decoder = tf.keras.layers.Input(shape = (Decoder_Input_train.shape[-1]))
embedding_2 = tf.keras.layers.Embedding(EngWord+1, 256)(input_decoder)
output, _,  _ = tf.keras.layers.LSTM(256, return_sequences = True, return_state = True)(embedding_2, initial_state = encoder_state)
output = tf.keras.layers.Dense(TurWord+1, activation = "softmax")(output)
#############################################################################

model = tf.keras.models.Model([input_encoder, input_decoder], output)


model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


if __name__ == "__main__":
    model.fit([Encoder_Input_train, Decoder_Input_train], Decoder_Output_train, epochs = 50, callbacks = [callback], validation_split = 0.1)

