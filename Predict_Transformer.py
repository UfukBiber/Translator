import tensorflow as tf
import numpy as np
import EncoderDecoder as m

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)])
  except RuntimeError as e:
    print(e)



class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxWord, embeddingDims, seqLength, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.maxWord = maxWord
        self.embeddingDims = embeddingDims
        self.seqLegth = seqLength
        self.embeddingWord = tf.keras.layers.Embedding(maxWord, embeddingDims)
        self.embeddingPosition = tf.keras.layers.Embedding(seqLength, embeddingDims)
    
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        position = tf.range(start = 0, limit = length, delta = 1)
        embeddedTokens = self.embeddingWord(inputs)
        embeddedPos = self.embeddingPosition(position)
        return embeddedTokens + embeddedPos
    
    def compute_mask(self, inputs, mask = None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxWord":self.maxWord,
            "embeddingDims":self.embeddingDims,
            "seqLength":self.seqLegth,
        })
        return config
    
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embedDims, numHeads, denseDims, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
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
    def call(self, inputs, mask = None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attentionOutput = self.attention(inputs, inputs, attention_mask = mask)
        projInput = self.layerNormalization1(attentionOutput)
        projOut = self.denseProj(projInput)
        return self.layerNormalization2(projInput + projOut)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedDims":self.embedDims,
            "numHeads":self.numHeads,
            "denseDims":self.denseDims,
        })
        return config
    
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, embedDims, numHeads, denseDims, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
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
        self.support_masking = True
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
        mask = tf.expand_dims(mask, axis = 0)
        mult = tf.concat([tf.expand_dims(batchSize, -1), tf.constant([1, 1], dtype = tf.int32)], axis = 0)
        return tf.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embedDims":self.embedDims,
            "numHeads":self.numHeads,
            "denseDims":self.denseDims,
        })
        return config

    
model = tf.keras.models.load_model("TransformerModel", custom_objects={
    "PositionalEmbedding":PositionalEmbedding,
    "TransformerEncoder":TransformerEncoder,
    "TransformerDecoder":TransformerDecoder,
})

string = "I am very happy because schools are off today."
string_2 = "Reading is a very nice habit"
string_3 = "My friend is very angry to me."



a = "[START]"
b = m.TarVectorization([a])[:, :-1]
print(b)

def Predict(text):
    InpVector = m.InpVectorization([text])
    decodedSentence = "[START]"
    step = 0
    Result =  []
    output = 0
    while output != 3 and step <= 20:
        tarVector = m.TarVectorization([decodedSentence])[:, :-1]
        output = model.predict([InpVector, tarVector])
        output = np.argmax(output[0, step, :])
        decodedSentence += " " + m.TarVocabularies[output]
        step += 1
    return decodedSentence

print(Predict(string))
print(Predict(string_2))
print(Predict(string_3))