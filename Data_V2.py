import re
import numpy as np

t = 3
e = 3

Tur2Number = {"_" : 0, "/s" : 1, "/e" : 2}
Eng2Number = {"_" : 0, "/s" : 1, "/e" : 2}
Number2Tur = {0 : "_", 1 : "/s", 2 : "/e"}
Number2Eng = {0 : "_", 1 : "/s", 2 : "/e"}

MaxLenTur = 0
MaxLenEng = 0

def PrepareData(directory):
    global e, t
    EncoderInput = []
    Decoder_For_Input = []
    Decoder_For_Output = []
    Decoder_Back_Input = []
    Decoder_Back_Output = []
    with open(directory, "r") as f:
        data = f.readlines()
        f.close()
    data = data[0:50000]
    for line in data:
        line = re.split(r"\t", line)[0:2]
        EncoderInput.append(VectorizeData(line[0], False))
        tur, tur1, reversed, reversed_1= VectorizeData(line[1], True)
        Decoder_For_Input.append(tur)
        Decoder_For_Output.append(tur1)
        Decoder_Back_Input.append(reversed)
        Decoder_Back_Output.append(reversed_1)
    for i in range(len(Decoder_For_Input)):
        if len(EncoderInput[i]) < MaxLenEng:
            for j in range(MaxLenEng-len(EncoderInput[i])):
                EncoderInput[i].append(0)

        if len(Decoder_For_Input[i]) < MaxLenTur:
            for j in range(MaxLenTur - len(Decoder_For_Input[i])):
                Decoder_For_Input[i].append(0)

        if len(Decoder_For_Output[i]) < MaxLenTur:
            for j in range(MaxLenTur - len(Decoder_For_Output[i])):
                Decoder_For_Output[i].append(0)

        if len(Decoder_Back_Input[i]) < MaxLenTur:
            for j in range(MaxLenTur - len(Decoder_Back_Input[i])):
                Decoder_Back_Input[i].append(0)

        if len(Decoder_Back_Output[i]) < MaxLenTur:
            for j in range(MaxLenTur - len(Decoder_Back_Output[i])):
                Decoder_Back_Output[i].append(0)
    EncoderInput = np.asarray(EncoderInput)
    Decoder_For_Input = np.asarray(Decoder_For_Input)
    Decoder_For_Output = np.asarray(Decoder_For_Output)
    Decoder_Back_Input = np.asarray(Decoder_Back_Input)
    Decoder_Back_Output = np.asarray(Decoder_Back_Output)
    return EncoderInput, Decoder_For_Input, Decoder_For_Output, Decoder_Back_Input

def VectorizeData(sentence, isTurkish):  
    global e, t
    global Tur2Number, Eng2Number
    global MaxLenTur, MaxLenEng
    sentence = re.sub(r"\W+", " ", sentence).lower()
    words =  re.split(" ", sentence)
    tokens = []
    if  isTurkish:
        tokens_1 = []
        for word in words:
            token = Tur2Number.get(word, t)
            tokens.append(token)
            tokens_1.append(token)
            if token == t:
                Tur2Number[word] = t
                Number2Tur[t] = word
                t += 1
        reversedTokens = tokens.copy()
        reversedTokens.reverse()
        reversedTokens_1 = reversedTokens.copy()
        tokens.append(2)
        tokens.insert(0, 1)
        tokens_1.append(2)
        reversedTokens.append(1)
        reversedTokens.insert(0,2)
        reversedTokens_1.append(1)
        if MaxLenTur < len(tokens):
            MaxLenTur = len(tokens)
        return tokens, tokens_1, reversedTokens, reversedTokens_1
    else:
        for word in words:
            token = Eng2Number.get(word, e)
            tokens.append(token)
            if token == e:
                Eng2Number[word] = e
                Number2Eng[e] = word
                e += 1
        tokens.insert(0, 1)
        tokens.append(2)
        if MaxLenEng < len(tokens):
            MaxLenEng = len(tokens)
        return tokens





