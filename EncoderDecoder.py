import tensorflow as tf 
import numpy as np
import string, re
import random


VALIDATION_SPLIT = 0.1
BATCH_SIZE = 32

def get_pairs(dir):
    pairs = []
    with open(dir, "r") as f:
        for line in f:
            inp, tar, _ = line.split("\t")
            pairs.append((inp, tar))
        f.close()
    return pairs 

def divide_pairs(pairs):
    Inp = [pair[0] for pair in pairs]
    Tar = [pair[1] for pair in pairs]
    return Inp, Tar


pairs = get_pairs("tur.txt")
TRAIN_LENGTH = int(len(pairs) * (1 - VALIDATION_SPLIT))

random.seed(23412)
random.shuffle(pairs)

trainPairs = pairs[:TRAIN_LENGTH]
valPairs = pairs[TRAIN_LENGTH:]





trainInp, trainTar = divide_pairs(trainPairs)
valInp, valTar = divide_pairs(valPairs)

def tar_custom_stardardization(input_string):
    input_string = tf.strings.lower(input_string)
    input_string = tf.strings.regex_replace(input_string, f"[{re.escape(string.punctuation)}]", "")
    input_string = tf.strings.join(["[START]", input_string, "[END]"],  " ")
    return input_string

InpVectorization = tf.keras.layers.TextVectorization(output_mode = "int",
                                                    output_sequence_length = 20,
                                                    max_tokens = 15000)
TarVectorization = tf.keras.layers.TextVectorization(output_mode = "int",
                                                     standardize = tar_custom_stardardization,
                                                     output_sequence_length = 21,
                                                     max_tokens = 15000)

InpVectorization.adapt(trainInp)
TarVectorization.adapt(trainTar)

InpVocabularies = InpVectorization.get_vocabulary()
TarVocabularies = TarVectorization.get_vocabulary()

trainDataset = tf.data.Dataset.from_tensor_slices((trainInp, trainTar)).batch(BATCH_SIZE)
valDataset = tf.data.Dataset.from_tensor_slices((valInp, valTar)).batch(BATCH_SIZE)


def format_data(inp, tar):
    inp = InpVectorization(inp)
    tar = TarVectorization(tar)
    return ({"Encoder_Input":inp, "Decoder_Input":tar[:, :-1]}, tar[:, 1:])

trainDataset = trainDataset.map(format_data, num_parallel_calls=4)
valDataset = valDataset.map(format_data, num_parallel_calls=4)
trainDataset = trainDataset.shuffle(2048).prefetch(16).cache()
valDataset = valDataset.shuffle(2048).prefetch(16).cache()


encoderInput = tf.keras.layers.Input(shape = (None, ), name = "Encoder_Input")
embedded = tf.keras.layers.Embedding(len(InpVocabularies), 256, name = "Encoder_Embedding", mask_zero = True)(encoderInput)
_, forward_state_h, forward_state_c, backward_state_h, backward_state_c = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_state = True, name = "Encoder_LSTM"))(embedded)
encoder_state_h = tf.keras.layers.Concatenate()([forward_state_h, backward_state_h])
encoder_state_c = tf.keras.layers.Concatenate()([forward_state_c, backward_state_c])
encoderStates = [encoder_state_h, encoder_state_c]


# #############################################################################################

# # ######################## DECODER  ##############################################

decoderInput = tf.keras.layers.Input(shape = (None, ), name = "Decoder_Input")
output = tf.keras.layers.Embedding(len(TarVocabularies), 256, name = "Decoder_Embedding", mask_zero = True)(decoderInput)
output, _,  _ = tf.keras.layers.LSTM(512, return_sequences = True, return_state = True, name = "Decoder_LSTM")(output, initial_state = encoderStates)
output = tf.keras.layers.Dropout(0.5)(output)
output = tf.keras.layers.Dense(len(TarVocabularies), activation = "softmax")(output)

if __name__ == "__main__":
    # model = tf.keras.models.Model([encoderInput, decoderInput], output)
    model = tf.keras.models.load_model("EncoderDecoderModel")
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    model.fit(trainDataset, validation_data = valDataset, epochs = 3, callbacks = [tf.keras.callbacks.ModelCheckpoint("EncoderDecoderModel", save_best_only = True)])