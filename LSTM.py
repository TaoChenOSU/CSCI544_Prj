# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical


## Plot
# import plotly.offline as py
# import plotly.graph_objs as go
# py.init_notebook_mode(connected=True)
# import matplotlib as plt

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Other
import re
import string
import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle
from sklearn.manifold import TSNE


def create_conv_model(vocabulary_size, lyrics_maxlen):
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, 300, input_length=lyrics_maxlen))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=2))
    model_conv.add(LSTM(300))
    model_conv.add(Dense(9, activation='softmax'))
    model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

if __name__ == "__main__":
    STEM = False
    TO_SEQ = True
    TO_MATRIX_TFIDF = False
    PRE_TRAIN_EBD = False
    # load data from disk
    with open("testNB_lyrics", "rb") as lyrics_file, open("testNB_labels", "rb") as labels_file:
        lyrics = pickle.load(lyrics_file)
        labels = pickle.load(labels_file)
    # split training and test set.
    # last 300 used as test, others used as training
    lyrics_shuffle, labels_shuffle = shuffle(lyrics, labels, random_state=0)
    labels_shuffle = to_categorical(labels_shuffle)
    # print(labels_shuffle.shape)

    if STEM:
        ########## TODO: implement this. Add stemming preprocessor. Maybe also the stop word remover ########
        pass

    if TO_SEQ:
        vocabulary_size = 20000
        lyrics_maxlen = 400
        tokenizer = Tokenizer(num_words= vocabulary_size)
        tokenizer.fit_on_texts(lyrics_shuffle)
        # print ("num of unique words:", len(tokenizer.word_counts))
        sequences = tokenizer.texts_to_sequences(lyrics_shuffle)
        data = pad_sequences(sequences, maxlen=lyrics_maxlen)

    if TO_MATRIX_TFIDF:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lyrics_shuffle)
        encoded = tokenizer.texts_to_matrix(lyrics_shuffle, mode='tfidf')
        vocabulary_size = len(tokenizer.word_counts)
        data = encoded

    if PRE_TRAIN_EBD:
        ########## TODO: load pre-trained glove embedding ########
        pass
    # print ("data shape:", data.shape)
    lyrics_train = data[:-300]
    labels_train = np.array(labels_shuffle[:-300])
    lyrics_test = data[-300:]
    labels_test = np.array(labels_shuffle[-300:])

    model_conv = create_conv_model(vocabulary_size, lyrics_maxlen)
    model_conv.fit(lyrics_train, labels_train, validation_split=0.1, epochs = 10)

    scores = model_conv.evaluate(lyrics_test, labels_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
