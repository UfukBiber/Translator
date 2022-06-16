import tensorflow as tf 
import string, re
import random
import numpy as np


# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
#   except RuntimeError as e:
#     print(e)

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
                                                    output_sequence_length = SEQEUNCE_LENGTH,
                                                    max_tokens = NUMBER_OF_WORDS)
TarVectorization = tf.keras.layers.TextVectorization(output_mode = "int",
                                                     standardize = tar_custom_stardardization,
                                                     output_sequence_length = SEQEUNCE_LENGTH+1,
                                                     max_tokens = NUMBER_OF_WORDS)

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
trainDataset = trainDataset.shuffle(2048).prefetch(16)
valDataset = valDataset.shuffle(2048).prefetch(16)




class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxWord, embeddingDims, seqLength, **kwargs):
        self.maxWord = maxWord
        self.embeddingDims = embeddingDims
        self.seqLegth = seqLength
        self.embeddingWord = tf.keras.layers.Embedding(maxWord, embeddingDims)
        self.embeddingPosition = tf.keras.layers.Embedding(seqLength, embeddingDims)
        super(PositionalEmbedding, self).__init__(**kwargs)
    
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        position = tf.range(start = 0, limit = length, delta = 1)
        embeddedTokens = self.embeddingWord(inputs)
        embeddedPos = self.embeddingPosition(position)
        return embeddedTokens + embeddedPos
    
    def compute_mask(self, inputs, mask = None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "maxWord":self.maxWord,
            "embeddingDims":self.embeddingDims,
            "seqLength":self.seqLegth,
        })
        return config
    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embedDims, numHeads, denseDims, **kwargs):
        self.embedDims = embedDims
        self.numHeads = numHeads
        self.denseDims = denseDims

        self.attention = tf.keras.layers.MultiHeadAttention(numHeads, embedDims)
        self.denseProj = tf.keras.Sequential([
            tf.keras.layers.Dense(denseDims, activation = "relu"),
            tf.keras.layers.Dense(embedDims)
        ])
        self.layerNormalization1 = tf.keras.layers.LayerNormalization() 
        self.layerNormalization2 = tf.keras.layers.LayerNormalization() 
        super(TransformerEncoder, self).__init__(**kwargs)
        self.supports_masking = True
    def call(self, inputs, mask = None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attentionOutput = self.attention(inputs, inputs, attention_mask = mask)
        projInput = self.layerNormalization1(inputs + attentionOutput)
        projOut = self.denseProj(projInput)
        return self.layerNormalization2(projInput + projOut)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "embedDims":self.embedDims,
            "numHeads":self.numHeads,
            "denseDims":self.denseDims,
        })
        return config
    
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embedDims, numHeads, denseDims, **kwargs):
        self.embedDims = embedDims
        self.numHeads = numHeads
        self.denseDims = denseDims

        self.attention1 = tf.keras.layers.MultiHeadAttention(numHeads, embedDims)
        self.attention2 = tf.keras.layers.MultiHeadAttention(numHeads, embedDims)
        self.denseProj = tf.keras.Sequential([
            tf.keras.layers.Dense(denseDims, activation = "relu"),
            tf.keras.layers.Dense(embedDims)
        ])
        self.layerNormalization1 = tf.keras.layers.LayerNormalization() 
        self.layerNormalization2 = tf.keras.layers.LayerNormalization() 
        self.layerNormalization3 = tf.keras.layers.LayerNormalization() 
        super(TransformerDecoder, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, encoderOut, mask = None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            mask = tf.cast(mask[:, tf.newaxis, :], dtype = tf.int32)
            mask = tf.minimum(mask, causal_mask)
        attentionOut_1 = self.attention1(query = inputs, value = inputs, key = inputs, attention_mask = causal_mask)
        attentionOut_1 = self.layerNormalization1(inputs + attentionOut_1)
        attentionOut_2 = self.attention2(query = attentionOut_1, value = encoderOut, key = encoderOut, attention_mask = mask)
        attentionOut_2 = self.layerNormalization2(attentionOut_1 + attentionOut_2)
        projOut = self.denseProj(attentionOut_2)
        return self.layerNormalization3(projOut)
    
    def get_causal_attention_mask(self, inputs):
        inputShape = tf.shape(inputs)
        batchSize, seqLength = inputShape[0], inputShape[1]
        i = tf.range(seqLength)[:, tf.newaxis]
        j = tf.range(seqLength)
        mask = tf.cast(i >= j, dtype = tf.int32)
        mask = tf.reshape(mask, (1, inputShape[1], inputShape[1]))
        mult = tf.concat([tf.expand_dims(batchSize, -1), 
                          tf.constant([1, 1], dtype = tf.int32)], axis = 0)
        return tf.tile(mask, mult)

    def get_config(self):
        config = super(TransformerDecoder, self).get_config()
        config.update({
            "embedDims":self.embedDims,
            "numHeads":self.numHeads,
            "denseDims":self.denseDims,
        })
        return config

encoderInp = tf.keras.layers.Input(shape = (None, ), name = "Encoder_Input")
encoderEmbedding = PositionalEmbedding(len(InpVocabularies), EMBEDDING_DIMS, SEQEUNCE_LENGTH)(encoderInp)
encoderOut = TransformerEncoder(EMBEDDING_DIMS, NUM_HEADS, DENSE_DIMS)(encoderEmbedding)

decoderInp = tf.keras.layers.Input(shape = (None, ), name = "Decoder_Input")
decoderEmbedding = PositionalEmbedding(len(TarVocabularies), EMBEDDING_DIMS, SEQEUNCE_LENGTH)(decoderInp)
decoderOut = TransformerDecoder(EMBEDDING_DIMS, NUM_HEADS, DENSE_DIMS)(decoderEmbedding, encoderOut)
decoderOut = tf.keras.layers.Dropout(0.5)(decoderOut)
decoderOut = tf.keras.layers.Dense(len(TarVocabularies), activation = "softmax")(decoderOut)

model = tf.keras.models.Model([encoderInp, decoderInp], decoderOut)

model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

model = tf.keras.models.load_model("TransformerModel", custom_objects={
    "PositionalEmbedding":PositionalEmbedding,
    "TransformerEncoder":TransformerEncoder,
    "TransformerDecoder":TransformerDecoder,
})

model.fit(trainDataset, validation_data = valDataset, epochs =3 , callbacks = [tf.keras.callbacks.ModelCheckpoint("TransformerModel", monitor = "val_accuracy", save_best_only = True),
                                                                                tf.keras.callbacks.EarlyStopping(monitor = "val_accuracy", patience = 2)])


string_1 = "I am very happy because schools are off today."
string_2 = "Reading is a very nice habit"
string_3 = "My friend is very angry to me."


def Predict(text):
    InpVector = InpVectorization([text])
    decodedSentence = "[START]"
    step = 0
    Result =  []
    output = 0
    while output != 3 and step <= 20:
        tarVector = TarVectorization([decodedSentence])[:, :-1]
        output = model.predict([InpVector, tarVector])
        output = np.argmax(output[0, step, :])
        decodedSentence += " " + TarVocabularies[output]
        step += 1
    return decodedSentence

print(Predict(string_1))
print(Predict(string_2))
print(Predict(string_3))