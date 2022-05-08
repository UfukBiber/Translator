import tensorflow as tf 
import Data
import numpy as np


NumOfSamples = 5000



model = tf.keras.models.load_model("my_model_deu")
decInp, decOut, encInp = Data.PrepareData("deu.txt")

Num2Tur = Data.Number2Tur
Num2Eng = Data.Number2Eng



decInp_test = decInp[:NumOfSamples]
decOut_test = decOut[:NumOfSamples]
encInp_test = encInp[:NumOfSamples]

model.evaluate([encInp_test, decInp_test], decOut_test)