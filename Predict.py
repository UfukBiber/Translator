import tensorflow as tf
import Model_3 as m3
import numpy as np
#### 2 is [START]
#### 3 is [END]
model = tf.keras.models.load_model("my_model_tur")

###########################ENCODER#############

EncoderInput = model.inputs[0]
output, for_state_h, for_state_c, back_state_h, back_state_c = model.layers[3].output

encoder_state_h = tf.concat([for_state_h, back_state_h], axis = -1)
encoder_state_c = tf.concat([for_state_c, back_state_c], axis = -1)

encoderStates = [encoder_state_h, encoder_state_c]

encoderModel = tf.keras.models.Model(EncoderInput, encoderStates)


#####################################################

############################# DECODER #############

DecoderInput = model.input[1]
EmbedOut = model.layers[4](DecoderInput)
DecoderInpStates = [tf.keras.layers.Input(shape = (512,)), tf.keras.layers.Input(shape = (512, ))]

output, state_h, state_c = model.layers[7](EmbedOut, initial_state = DecoderInpStates)
output = model.layers[8](output)
output = model.layers[9](output)

DecoderOutStates = [state_h, state_c]
DecoderModel = tf.keras.models.Model([DecoderInput]+DecoderInpStates, [output]+DecoderOutStates)

#######################################################3

InpVectorization = m3.InpVectorization
wordsInp = m3.InpWords
TarVectorization = m3.TarVectorization
TarWords = m3.TarWords
string = "I am very happy"




def Predict(text):
    InpVector = InpVectorization([text])
    States = encoderModel(InpVector)
    TarInp = np.zeros((1, 1))
    TarInp[:, 0] = 2
    step = 0
    Result =  []
    output = 0
    while output != 3 and step <= 21:
        output, state_h, state_c = DecoderModel([TarInp] + States)
        States = [state_h, state_c]
        output = np.argmax(np.squeeze(output))
        TarInp[:, 0] = output
        Result.append(TarWords[output])
        step += 1
    String = " ".join(Result)
    return String

print(Predict(string))
