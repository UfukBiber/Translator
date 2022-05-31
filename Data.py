import tensorflow as tf
import re, string 
import matplotlib.pyplot as plt


def custom_standardize(text):
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, f"[{re.escape(string.punctuation)}]", "")
    text = tf.strings.join(["[START]", text, "[END]"], separator = " ")
    return text

class Data:
    def __init__(self, directory, MaxVocabSize = None, MaxSequenceLength = None):
        self.pair = self.read_data(directory)
        self.Inp, self.Tar = self.divide_pairs(self.pair)
        self.InpVectorizator = tf.keras.layers.TextVectorization(max_tokens = MaxVocabSize,
                                                                 output_mode = "int",
                                                                 output_sequence_length = MaxSequenceLength)
        self.TarVectorizator = tf.keras.layers.TextVectorization(max_tokens = MaxVocabSize,
                                                                 output_mode = "int",
                                                                 output_sequence_length = MaxSequenceLength + 1,
                                                                 standardize = custom_standardize)

    def read_data(self, directory):
        data = []
        with open(directory, "r") as file:
            lines = file.readlines()
        for line in lines:
            line = re.split("\t", line)
            data.append([line[0], line[1]])
        return data 
    def divide_pairs(self, pair):
        Inp, Tar = zip(*pair)
        return Inp, Tar