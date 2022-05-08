import tensorflow as tf 
import Data
import numpy as np




numberOfSamples = 5000
model = tf.keras.models.load_model("my_model_tur")

decInp, decOut, encInp = Data.PrepareData("tur.txt")

decInp, decOut, encInp = decInp[:numberOfSamples], decOut[:numberOfSamples], encInp[:numberOfSamples]

Word2Token = Data.Tur2Number
Token2Word = Data.Number2Tur
decInpLength = Data.MaxLenTur
encInpLength = Data.MaxLenEng

####################### Encoder #############################
Enc_Inp = model.inputs[0]
Enc_Out, Enc_state_h, Enc_state_c = model.layers[4].output
Enc_states = [Enc_state_h, Enc_state_c] 

Enc_model = tf.keras.models.Model(Enc_Inp, Enc_states)

####################### Decoder ##############################

Dec_Inp = model.inputs[1]
embed_Dec = model.layers[3].output
Dec_Inp_state_h, Dec_Inp_state_c = tf.keras.layers.Input(shape = (256, ), name = "input_3"), tf.keras.layers.Input(shape = (256, ), name = "input_4")
Dec_Inp_states = [Dec_Inp_state_h, Dec_Inp_state_c]
Dec_out, Dec_Out_state_h, Dec_Out_state_c = model.layers[-2](embed_Dec, initial_state = Dec_Inp_states)
Dec_Out_states = [Dec_Out_state_h,Dec_Out_state_c]
Dec_out = model.layers[-1](Dec_out)

Dec_model = tf.keras.models.Model([Dec_Inp] + Dec_Inp_states, [Dec_out] + Dec_Out_states)



def Predict(ind):
    enc_Inp = np.expand_dims(encInp[ind], axis = 0)
    dec_Out = np.expand_dims(decOut[ind], axis = 0)
    initialStates = Enc_model(enc_Inp)
    stop = False
    dec_Inp = np.zeros((1, 1))
    dec_Inp[:, 0] = Word2Token["/s"]
    step = 1
    sentence = []
    while not stop:
        output, h, c = Dec_model([dec_Inp] + initialStates)
        output = np.argmax(np.squeeze(output))
        initialStates = [h, c]
        dec_Inp[:, 0] = output
        sentence.append(output)
        step += 1
        if step == 18 or output == Word2Token["/e"]:
            break
    return sentence, dec_Out



def decodeSentence(sentence):
    if type(sentence) is not list and len(sentence.shape)!= 1:
        sentence = np.squeeze(sentence)
    Sentence = ""
    for token in sentence:
        if Token2Word[token] == "/e" or Token2Word[token] == "_":
            break
        Sentence += Token2Word[token]
        Sentence += " "
    return Sentence

for i in range(30):
    predicted, real = Predict(i)
    print("Predicted : " , decodeSentence(predicted))
    print("Real Sentence : ", decodeSentence(real))
    print("\n")





