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
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.manifold import TSNE


def clean_text(text):

    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    # text = re.sub(r"what's", "what is ", text)
    # text = re.sub(r"\'s", " ", text)
    # text = re.sub(r"\'ve", " have ", text)
    # text = re.sub(r"n't", " not ", text)
    # text = re.sub(r"i'm", "i am ", text)
    # text = re.sub(r"\'re", " are ", text)
    # text = re.sub(r"\'d", " would ", text)
    # text = re.sub(r"\'ll", " will ", text)
    # text = re.sub(r",", " ", text)
    # text = re.sub(r"\.", " ", text)
    # text = re.sub(r"!", " ! ", text)
    # text = re.sub(r"\/", " ", text)
    # text = re.sub(r"\^", " ^ ", text)
    # text = re.sub(r"\+", " + ", text)
    # text = re.sub(r"\-", " - ", text)
    # text = re.sub(r"\=", " = ", text)
    # text = re.sub(r"'", " ", text)
    # text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    # text = re.sub(r":", " : ", text)
    # text = re.sub(r" e g ", " eg ", text)
    # text = re.sub(r" b g ", " bg ", text)
    # text = re.sub(r" u s ", " american ", text)
    # text = re.sub(r"\0s", "0", text)
    # text = re.sub(r" 9 11 ", "911", text)
    # text = re.sub(r"e - mail", "email", text)
    # text = re.sub(r"j k", "jk", text)
    # text = re.sub(r"\s{2,}", " ", text)

    # text = text.split()
    # stemmer = SnowballStemmer('english')
    # stemmed_words = [stemmer.stem(word) for word in text]
    # text = " ".join(stemmed_words)

    return text


def load_glove_vectors(glove_path):
    embeddings_index = dict()
    f = open(glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    return embeddings_index


def create_conv_model(vocabulary_size,
                      lyrics_maxlen,
                      use_glove=False,
                      embedding_matrix=None):
    model_conv = Sequential()
    if use_glove:
        # "Freeze" the word embeddings by setting trainable=False
        model_conv.add(
            Embedding(
                vocabulary_size,
                300,
                input_length=lyrics_maxlen,
                weights=[embedding_matrix],
                trainable=False))
    else:
        model_conv.add(
            Embedding(vocabulary_size, 300, input_length=lyrics_maxlen))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(128, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(128, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(128, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # model_conv.add(Flatten())
    # model_conv.add(Dense(128, activation='relu'))
    model_conv.add(Dense(8, activation='softmax'))
    model_conv.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model_conv


def lstm_clf(lyrics_raw, labels_int):
    PRE_PROCESS = True
    TO_SEQ = True
    TO_MATRIX_TFIDF = False
    PRE_TRAIN_EBD = True

    if PRE_PROCESS:
        lyrics = list(map(clean_text, lyrics_raw))
        # print (lyrics[0])
    # lyrics_shuffle, labels_shuffle = shuffle(lyrics, labels, random_state=0)
    # print(labels_shuffle.shape)

    if TO_SEQ:
        vocabulary_size = 20000  # use top 20000 words in training set to create vocaulary
        lyrics_maxlen = 300  # keep 300 words in each lyrics. truncating or padding zeros
        tokenizer = Tokenizer(num_words=vocabulary_size)
        tokenizer.fit_on_texts(lyrics)
        # print ("num of unique words:", len(tokenizer.word_counts))
        # words in text are converted to int values, can check with tokenizer.word_index
        sequences = tokenizer.texts_to_sequences(lyrics)
        data = pad_sequences(sequences, maxlen=lyrics_maxlen)

    if TO_MATRIX_TFIDF:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lyrics)
        encoded = tokenizer.texts_to_matrix(lyrics, mode='tfidf')
        vocabulary_size = len(tokenizer.word_counts)
        data = encoded

    if PRE_TRAIN_EBD:
        ########## TODO: load pre-trained glove embedding ########
        embeddings_index = load_glove_vectors("./glove.6B/glove.6B.300d.txt")
        embedding_matrix = np.zeros((vocabulary_size, 300))
        for word, index in tokenizer.word_index.items():
            if index > vocabulary_size - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector
        print("Created embedding matrix.")
    # print ("data shape:", data.shape)

    # split training and test set.
    labels = to_categorical(np.array(labels_int))
    lyrics_train, lyrics_test, labels_train, labels_test = train_test_split(
        data,
        labels,
        test_size=0.1,
        random_state=7,
        shuffle=True,
        stratify=labels)

    model_conv = create_conv_model(
        vocabulary_size,
        lyrics_maxlen,
        use_glove=True,
        embedding_matrix=embedding_matrix)
    model_conv.fit(lyrics_train, labels_train, validation_split=0.1, epochs=6)

    pred = model_conv.predict_classes(lyrics_test)
    print("F1 micro:{}, F1 macro:{}, F1 weighted:{}".format(
        precision_recall_fscore_support(
            np.argmax(labels_test, axis=1), pred, average="micro")[2],
        precision_recall_fscore_support(
            np.argmax(labels_test, axis=1), pred, average="macro")[2],
        precision_recall_fscore_support(
            np.argmax(labels_test, axis=1), pred, average="weighted")[2]))
    scores = model_conv.evaluate(lyrics_test, labels_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))




if __name__ == "__main__":
    # load data from disk
    with open("test_lyrics_words", "rb") as lyrics_file, open(
            "test_labels", "rb") as labels_file:
        lyrics_raw = pickle.load(lyrics_file)
        labels_int = pickle.load(labels_file)
        # print(lyrics[0])
    lstm_clf(lyrics_raw, labels_int)