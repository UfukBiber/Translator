import tensorflow as tf 
import Data_V2
import numpy as np




numberOfSamples = 5000
model = tf.keras.models.load_model("my_model_tur_V2")
print(model.layers)

encInp, decForInp, decForOut, decBackInp = Data_V2.PrepareData("tur.txt")



Word2Token = Data_V2.Tur2Number
Token2Word = Data_V2.Number2Tur
decInpLength = Data_V2.MaxLenTur
encInpLength = Data_V2.MaxLenEng
LengthOfOutput = Data_V2.MaxLenTur
# ##################### Encoder #############################
Enc_Inp = model.inputs[0]
Enc_Out, F_E_S_h, F_E_S_c, B_E_S_h, B_E_S_c = model.layers[4].output
Enc_states_h = tf.concat([F_E_S_h, B_E_S_h], axis = -1)
Enc_states_c = tf.concat([F_E_S_c, B_E_S_c])
Enc_states = [Enc_states_h, Enc_states_c]

Enc_model = tf.keras.models.Model(Enc_Inp, Enc_states)



# ###################### Decoder ##############################





def Predict(ind):
    enc_Inp = np.expand_dims(encInp[ind], axis = 0)
    dec_For_Inp = np.expand_dims(decForInp[ind], axis = 0)
    dec_For_Out = np.expand_dims(decForOut[ind], axis = 0)
    dec_Back_Inp = np.expand_dims(decBackInp[ind], axis = 0)
    initialStates = Enc_model(enc_Inp)
    For_initialStates = initialStates[0]
    Back_initialStates = initialStates[1]
    stop = False
    dec_For_Inp = np.zeros((1, 1))
    dec_Back_Inp = np.zeros((1, 1))
    dec_For_Inp[:, 0] = 1
    dec_Back_Inp[:, 0] = 2
    Results = []
    while not stop:
        output, f_h, f_c, b_h, b_c = Dec_Model([dec_For_Inp, dec_Back_Inp]+For_initialStates+Back_initialStates)
        output = np.argmax(np.squeeze(output))
        For_initialStates = [f_h, f_c]
        Back_initialStates = [b_h, b_c]
        Results.append(output)

        if len(Results) == LengthOfOutput or output == 2:
            break
    return Results, dec_For_Out
y, real = Predict(10)

# print(y)
# print(real)

# # def decodeSentence(sentence):
# #     if type(sentence) is not list and len(sentence.shape)!= 1:
# #         sentence = np.squeeze(sentence)
# #     Sentence = ""
# #     for token in sentence:
# #         if Token2Word[token] == "/e" or Token2Word[token] == "_":
# #             break
# #         Sentence += Token2Word[token]
# #         Sentence += " "
# #     return Sentence

# # for i in range(10):
# #     predicted, real = Predict(i)
# #     print("Predicted : " , decodeSentence(predicted))
# #     print("Real Sentence : ", decodeSentence(real))
# #     print("\n")





