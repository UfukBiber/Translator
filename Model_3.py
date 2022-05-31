import tensorflow as tf
import re, string 
import random



MaxVocabSize = 15000
MaxSequenceLength = 20
ValidationSplit = 0.1

def readData(directory, Quantity = None):
    Data = []
    with open(directory, "r") as f:
        lines = f.readlines()
        f.close()
    if Quantity:
        lines = lines[:Quantity]
    for line in lines :
        line = re.split("\t", line)
        Data.append((line[0], line[1]))
    return Data

def dividePair(pairs):
    Eng, Tur = zip(*pairs)
    return Eng, Tur

def custom_standardize(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, f"[{re.escape(string.punctuation)}]", "")
    text = tf.strings.join(["[START]", text, "[END]"], separator = " ")
    return text


InpVectorization = tf.keras.layers.TextVectorization(max_tokens = MaxVocabSize,
                                                    output_mode = "int",
                                                    output_sequence_length = MaxSequenceLength)

TarVectorization = tf.keras.layers.TextVectorization(max_tokens = MaxVocabSize,
                                                    output_mode = "int",
                                                    output_sequence_length = MaxSequenceLength + 1,
                                                    standardize = custom_standardize)



Pairs = readData("tur.txt")
TrainLen = (len(Pairs) // 10) * 9

TrainPairs = Pairs[:TrainLen]


Inp, Tar = dividePair(Pairs)
TrainInp, TrainTar = dividePair(TrainPairs)


InpVectorization.adapt(Inp)
TarVectorization.adapt(Tar)




def getData(Inp, Tar):
    Inp = InpVectorization(Inp)
    Tar = TarVectorization(Tar)
    DecoderInput = Tar[:, :-1]
    Output = Tar[:, 1:]
    return Inp, DecoderInput, Output



InpWords = InpVectorization.get_vocabulary()
TarWords = TarVectorization.get_vocabulary()

Train_EncoderInput, Train_DecoderInput, Train_Output = getData(TrainInp, TrainTar)



def formatDataset(Inp, Tar):
    return ({"EncoderInput":Inp, "DecoderInput":Tar[:, :-1]}, Tar[:, 1:]) 



# class MyCallBack(tf.keras.callbacks.Callback):
#     def __init__(self):
#         self.acc = 0.80
#     def on_epoch_end(self, epoch, logs={}):
#         if logs.get("accuracy") > self.acc:
#             print("\n\nReached %f percent accuracy"%self.acc)
#             print("Model is saved")
#             self.acc += 0.05
#             self.model.save("my_model_tur_V2")
#         if logs.get("accuracy")>0.98:
#             self.model.stop_training = True
#             self.model.save("my_model_tur_V2")
            

# callback = MyCallBack()

# ##########################ENCODER###########################################
# Encoder_Input = tf.keras.layers.Input(shape = (20, ), name = "Encoder_Input")
# Positional_Encoder_Input = tf.keras.layers.Input(shape = (20, ), name="Positional_Encoder_Input")
# embedded = tf.keras.layers.Embedding(len(InpWords), 256, name = "Encoder_Embedding", mask_zero = True)(Encoder_Input)
# _, forward_state_h, forward_state_c, backward_state_h, backward_state_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_state = True, name = "LSTM_1"))(embedded)
# encoder_state_h = tf.keras.layers.Concatenate()([forward_state_h, backward_state_h])
# encoder_state_c = tf.keras.layers.Concatenate()([forward_state_c, backward_state_c])
# encoderStates = [encoder_state_h, encoder_state_c]


# # #############################################################################################

# # # ######################## DECODER  ##############################################

# input_decoder = tf.keras.layers.Input(shape = (None, ), name = "input_decoder")
# output = tf.keras.layers.Embedding(len(TarWords), 256, name = "Embedding_2", mask_zero = True)(input_decoder)
# output, _,  _ = tf.keras.layers.LSTM(512, return_sequences = True, return_state = True, name = "LSTM_2")(output, initial_state = encoderStates)
# output = tf.keras.layers.Dropout(0.5)(output)
# output = tf.keras.layers.Dense(len(TarWords), activation = "softmax")(output)


# model = tf.keras.models.Model([Encoder_Input, Positional_Encoder_Input, input_decoder], output)

# model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])


# if __name__ == "__main__":
    
#     model.fit([Train_EncoderInput,  Train_DecoderInput], Train_Output , epochs = 50, callbacks = [callback], validation_split = ValidationSplit)
