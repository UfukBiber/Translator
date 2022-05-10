import tensorflow as tf 
import Data
import numpy as np





model = tf.keras.models.load_model("my_model_tur")
encInp, decInp, decOut = Data.PrepareData("tur.txt")

encInp = encInp[45000:]
decInp = decInp[45000:]
decOut = decOut[45000:]



model.evaluate([encInp, decInp], decOut)
