from regex import W
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



Pairs = readData("tur.txt", 6400)


Inp, Tar = dividePair(Pairs)

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

EncoderInput, DecoderInput, Output = getData(Inp, Tar)


class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self):
        self.acc = 0.5
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

class Encoder(tf.keras.layers.Layer):
    def __init__(self, Words, embedDims, GruUnits):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(Words, embedDims, mask_zero = True)
        self.GRU = tf.keras.layers.GRU(GruUnits, return_state = True, return_sequences = True)
    
    def call(self, inputs):
        embedded = self.embedding(inputs)
        gruOutput, gru_states = self.GRU(embedded)
        return gruOutput, gru_states

class Attention(tf.keras.layers.Layer):
    def __init__(self, Units):
        super(Attention, self).__init__()
        self.attention = tf.keras.layers.AdditiveAttention()
        self.W1 = tf.keras.layers.Dense(Units, activation = "relu", use_bias = False)
        self.W2 = tf.keras.layers.Dense(Units, activation = "relu", use_bias = False)

    def call(self, inputs, encoder_output):
        query = self.W1(inputs)
        key  = self.W2(encoder_output)
        content_vector = self.attention([query, encoder_output, key])
        return content_vector

class Decoder(tf.keras.layers.Layer):
    def __init__(self, Words, embedDims, Units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(Words, embedDims, mask_zero = True)
        self.GRU = tf.keras.layers.GRU(Units, return_sequences = True, return_state = True)
        self.attention = Attention(Units)
        self.concatenate = tf.keras.layers.Concatenate()
        self.W1 = tf.keras.layers.Dense(Units, activation = "relu", use_bias = False)
        self.W2 = tf.keras.layers.Dense(Words, activation = "softmax")
    
    def call(self, inputs, encoderOutput, state = None):
        embedded = self.embedding(inputs)
        gruOut, state = self.GRU(embedded, initial_state = state)
        contentVector = self.attention(gruOut, encoderOutput)
        rnnContentVector = self.concatenate([gruOut, contentVector])
        w1 = self.W1(rnnContentVector)
        w2 = self.W2(w1)
        return w2, state

class Model(tf.keras.Model):
    def __init__(self, embedDims, Units, InputVectorization, TarVectorization):
        super(Model, self).__init__()
        InpWordsSize = InputVectorization.vocabulary_size()
        TarWordsSize = TarVectorization.vocabulary_size()
        self.encoder = Encoder(InpWordsSize, embedDims, Units)
        self.decoder = Decoder(TarWordsSize, embedDims, Units)
        
    def train_step(self, inputs):
        EncInp = inputs[0][0]
        DecInp = inputs[0][1]
        Out = inputs[1]
        with tf.GradientTape() as tape:
            encOut, encState = self.encoder(EncInp)
            decState = encState
            loss = tf.constant(0.0)
            for i in range(DecInp.shape[1]):
                decOut, decState = self.decoder(tf.expand_dims(DecInp[:, i], -1), encOut, state = decState)
                loss = loss + self.compiled_loss(Out[:, i], decOut) / DecInp.shape[0]
            loss = loss / DecInp.shape[1]
            variables = self.trainable_variables 
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
        return {m.name: m.result() for m in self.metrics}



            
            
if __name__ == "__main__":
    model = Model(256, 1024, InpVectorization, TarVectorization)
    print(model.summary())
    # model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction = "sum"), optimizer = tf.keras.optimizers.Adam())
    # model.fit([EncoderInput, DecoderInput], Output, epochs = 5)
    # model.save("my_model_tur_V2")
    # encoder = Encoder(len(InpWords), 256, 128)
    # decoder = Decoder(len(TarWords), 256, 128)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction = "sum")
    # encOut, encState = encoder(EncoderInput)
    # decState = encState
    # loss = tf.constant(0.0)
    # for i in tf.range(20):
    #     decOut, decState = decoder(tf.expand_dims(DecoderInput[:, i], -1), encOut, state = decState)
    #     loss = loss + loss_fn(Output[:, i], decOut) / 64.0

    

    # print(loss / 20.0)
    
    
    