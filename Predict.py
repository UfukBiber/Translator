import tensorflow as tf
import numpy as np
import re, string
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 32
NUMBER_OF_WORDS = 12000
EMBEDDING_DIMS = 256
DENSE_DIMS = 2048
NUM_HEADS = 8
SEQEUNCE_LENGTH = 20

def get_pairs(dir):
    pairs = []
    with open(dir, "r") as f:
        for line in f:
            try:
                inp, tar, _ = line.split("\t")
            except:
                inp, tar = line.split("\t")
            pairs.append((inp, tar))
        f.close()
    return pairs 

def divide_pairs(pairs):
    Inp = [pair[0] for pair in pairs]
    Tar = [pair[1] for pair in pairs]
    return Inp, Tar

def tar_custom_stardardization(input_string):
    input_string = tf.strings.lower(input_string)
    input_string = tf.strings.regex_replace(input_string, f"[{re.escape(string.punctuation)}]", "")
    input_string = tf.strings.join(["[START]", input_string, "[END]"],  " ")
    return input_string

InpVectorization = tf.keras.layers.TextVectorization(output_mode = "int",
                                                    output_sequence_length = SEQEUNCE_LENGTH,
                                                    max_tokens = NUMBER_OF_WORDS)
TarVectorization = tf.keras.layers.TextVectorization(output_mode = "int",
                                                     standardize = tar_custom_stardardization,
                                                     output_sequence_length = SEQEUNCE_LENGTH+1,
                                                     max_tokens = NUMBER_OF_WORDS)

trainPairs = get_pairs("TrainPairs.txt")
TrainInp, TrainTar = divide_pairs(trainPairs)
    
InpVectorization.adapt(TrainInp)
TarVectorization.adapt(TrainTar)
InpVocabularies = InpVectorization.get_vocabulary()
TarVocabularies = TarVectorization.get_vocabulary()

model = tf.keras.models.load_model("EncoderDecoderModel")

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
        Result.append(TarVocabularies[output])
        step += 1
    String = " ".join(Result)
    return String
if __name__ == "__main__":
    string = "I am very happy because schools are off today."
    string_2 = "Reading is a very nice habit"
    string_3 = "My friend is very angry to me."
    print(Predict(string))
    print(Predict(string_2))
    print(Predict(string_3))