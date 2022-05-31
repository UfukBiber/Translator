import tensorflow as tf 
import Data
import numpy as np
import Data_2



####### Bot models  are at 80 % test accuracy
#  



EngVector, TurVector = Data_2.getVectors()

EncoderInp = EngVector
DecoderInp = TurVector[:, :-1]
DecoderOut = TurVector[:, 1:]

EncoderInpTest = EncoderInp[130000:]
DecoderInpTest = DecoderInp[130000:]
DecoderOutTest = DecoderOut[130000:]

model = tf.keras.models.load_model("my_model_tur")
# model_2 = tf.keras.models.load_model("my_model_tur_V2")


# encInp, decInp, decOut = Data.PrepareData("tur.txt", 50000)


# encInp = encInp[45000:]
# decInp = decInp[45000:]
# decOut = decOut[45000:]



model.evaluate([EncoderInpTest, DecoderInpTest], DecoderOutTest)
# model_2.evaluate([encInp, decInp], decOut)
