import tensorflow as tf 
import Data
import numpy as np




numberOfSamples = 5000
model = tf.keras.models.load_model("my_model_tur")
print(model.layers)

decInp, decOut, encInp = Data.PrepareData("tur.txt")

decInp, decOut, encInp = decInp[:numberOfSamples], decOut[:numberOfSamples], encInp[:numberOfSamples]

Word2Token = Data.Tur2Number
Token2Word = Data.Number2Tur
decInpLength = Data.MaxLenTur
encInpLength = Data.MaxLenEng

# ##################### Encoder #############################
# Enc_Inp = model.inputs[0]
# Enc_Out, F_E_S_h, F_E_S_c, B_E_S_h, B_E_S_c = model.layers[4].output
# Enc_For_states = [F_E_S_h, F_E_S_c]
# Enc_Back_states = [B_E_S_h, B_E_S_c]  

# Enc_model = tf.keras.models.Model(Enc_Inp, [Enc_For_states, Enc_Back_states])
# ###################### Decoder ##############################

# Dec_Inp = model.inputs[1]
# embed_Dec = model.layers[3].output
# F_D_I_S_h, F_D_I_S_c = tf.keras.layers.Input(shape = (256, ), name = "input_3"), tf.keras.layers.Input(shape = (256, ), name = "input_4")
# B_D_I_S_h, B_D_I_S_c = tf.keras.layers.Input(shape = (256, ), name = "input_5"), tf.keras.layers.Input(shape = (256, ), name = "input_6")

# Dec_LSTM_For = model.layers[5].forward_layer
# Dec_LSTM_Back = model.layers[5].backward_layer

# Dec_Inp_Forward_states = [F_D_I_S_h, F_D_I_S_c]
# Dec_Inp_Backward_states = [B_D_I_S_h, B_D_I_S_c]
# Dec_For_out, F_D_O_S_h, F_D_O_S_c= Dec_LSTM_For(embed_Dec, initial_state = Dec_Inp_Forward_states)
# Dec_Back_out, B_D_O_S_h, B_D_O_S_c = Dec_LSTM_Back(embed_Dec, initial_state = Dec_Inp_Backward_states)

# Dec_Out_For_states = [F_D_O_S_h, F_D_O_S_c]
# Dec_Out_Bac_states = [B_D_O_S_h, B_D_O_S_c]



# Dec_For_model = tf.keras.models.Model([Dec_Inp] + Dec_Inp_Forward_states, [Dec_For_out] + Dec_Out_For_states)
# Dec_Back_model = tf.keras.models.Model([Dec_Inp] + Dec_Inp_Backward_states, [Dec_Back_out] + Dec_Out_Bac_states)


# def Predict(ind):
#     enc_Inp = np.expand_dims(encInp[ind], axis = 0)
#     dec_Out = np.expand_dims(decOut[ind], axis = 0)
#     initialStates = Enc_model(enc_Inp)
#     For_initialStates = initialStates[0]
#     Back_initialStates = initialStates[1]
#     stop = False
#     dec_For_Inp = np.zeros((1, 1))
#     dec_Back_Inp = np.zeros((1, 1))
#     dec_For_Inp[:, 0] = 1
#     Forward_Results = []
#     Backward_Results = []
#     denseLayer = model.layers[-1]
#     while not stop:
#         for_output, f_h, f_c = Dec_For_model([dec_For_Inp] + For_initialStates)
#         back_output, b_h, b_c = Dec_Back_model([dec_Back_Inp] + Back_initialStates)
#         For_initialStates = [f_h, f_c]
#         Back_initialStates = [b_h, b_c]
#         Forward_Results.append(np.squeeze(for_output))
#         Backward_Results.insert(0, np.squeeze(back_output))
#         if len(Forward_Results) == decInpLength:
#             break
#     Forward_Results = np.asarray(Forward_Results)
#     Backward_Results = np.asarray(Backward_Results)
#     Result = np.concatenate((Forward_Results, Backward_Results), axis = 1)
#     Result = denseLayer(Result)
#     Result = np.argmax(np.squeeze(Result), axis = 1)
#     print(Result.shape)
#     return Result, dec_Out
# y, real = Predict(10)

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





