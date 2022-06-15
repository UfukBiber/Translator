import numpy as np
embedding_index = {}
with open("glove.6B.100d.txt") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "float64", sep = " ")
        embedding_index[word] = coefs
    f.close()

print(len(embedding_index.values()))