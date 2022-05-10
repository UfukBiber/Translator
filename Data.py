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
    DecoderInput = []
    DecoderOutput = []
    with open(directory, "r") as f:
        data = f.readlines()
        f.close()
    data = data[0:50000]
    for line in data:
        line = re.split(r"\t", line)[0:2]
        EncoderInput.append(VectorizeData(line[0], False))
        tur, tur1 = VectorizeData(line[1], True)
        DecoderInput.append(tur)
        DecoderOutput.append(tur1)
    for i in range(len(EncoderInput)):
        if len(EncoderInput[i]) < MaxLenEng:
            for _ in range(MaxLenEng-len(EncoderInput[i])):
                EncoderInput[i].append(0)
    for i in range(len(DecoderInput)):
        if len(DecoderOutput[i]) < MaxLenTur:
            for _ in range(MaxLenTur-len(DecoderInput[i])):
                DecoderInput[i].append(0)
            for _ in range(MaxLenTur-len(DecoderOutput[i])):
                DecoderOutput[i].append(0)
    return np.asarray(EncoderInput), np.asarray(DecoderInput), np.asarray(DecoderOutput)



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
        tokens.append(2)
        tokens.insert(0, 1)
        tokens_1.append(2)
        if MaxLenTur < len(tokens):
            MaxLenTur = len(tokens)
        return tokens, tokens_1
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






    
    
    

