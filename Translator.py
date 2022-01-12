import os
import numpy as np 
import re
class Data():
    def __init__(self):
        self.read_data()
    

    def read_data(self):
        eng, tur = [], []
        max_len_eng, max_len_tur = 0, 0
        with open("C:\\Users\\ufuk\\Desktop\\Translator\\tur.txt", "r", encoding = "UTF-8") as file:
            lines = file.readlines()
            file.close()
        for line in lines:
            line = re.split(r"\t", line)
            sentence = re.sub(r"[!.?;#$/,'\*\-\"]"," ", line[0]).lower().split()
            if len(sentence) > max_len_eng:
                max_len_eng = len(sentence)
            eng.append(sentence)
            sentence = re.sub(r"[!.:;?#$/,'\*\-\"]", " ", line[1]).lower().split()
            if len(sentence) > max_len_tur:
                max_len_tur = len(sentence)
            tur.append(sentence)
        return eng, tur, max_len_eng, max_len_tur
    def tokenizer(self):
        eng, tur, max_len_eng, max_len_tur = self.read_data()

if __name__ == "__main__" :
    Data = Data()
               
