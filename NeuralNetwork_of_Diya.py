import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

Stemmer = PorterStemmer()

def tokenize(text):
    return nltk.word_tokenize(text)

def stem(word):
    return Stemmer.stem(word.lower())

def word_list(tokenized_text,words):
    text_word=[stem(word) for word in tokenized_text]
    lists = np.zeros(len(words),dtype = np.float32)

    for id_no,wd in enumerate(words):
        if wd in text_word:
            lists[id_no] = 1
    return lists