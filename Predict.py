import tensorflow as tf
import numpy as np
import EncoderDecoder as m

model = tf.keras.models.load_model("EncoderDecoderModel")
print(model.layers)

###########################ENCODER#############

encoderInput = model.inputs[0]
encoderState_h = model.layers[5].output
encoderState_c = model.layers[6].output


encoderStates = [encoderState_h, encoderState_c]

encoderModel = tf.keras.models.Model(encoderInput, encoderStates)


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


string = "I am very happy because schools are off today."
string_2 = "Reading is a very nice habit"
string_3 = "My friend is very angry to me."




def Predict(text):
    InpVector = m.InpVectorization([text])
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
        Result.append(m.TarVocabularies[output])
        step += 1
    String = " ".join(Result)
    return String
if __name__ == "__main__":
    print(Predict(string))
    print(Predict(string_2))
    print(Predict(string_3))