import json
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import LSTM

def load_word2vec(filename):
    # Returns a dict containing a {word: numpy array for a dense word vector} mapping.
    # It loads everything into memory.
    
    w2vec={}
    with open(filename,"r") as f_in:
        for line in f_in:
            line_split=line.replace("\n","").split()
            w=line_split[0]
            vec=np.array([float(x) for x in line_split[1:]])
            w2vec[w]=vec
    return w2vec

word2vec_dict = load_word2vec("glove.6B/glove.6B.300d.txt")


def save_lyrics_label(lyrics_words_file, lyrics_embeddings_file, labels_file):
    """
    Save lyrics and labels as two lists. Then use pickle to save them to disk
    """

    genres = [
        "Hip-Hop", "Rock", "Pop", "Country", "Metal", "Holiday",
        "Children's Music", "Blues"
    ]
    genre2int = {
        "Blues": 0,
        "Hip-Hop": 1,
        "Rock": 2,
        "Pop": 3,
        "Country": 4,
        "Metal": 5,
        "Holiday": 6,
        "Children's Music": 7,
    }

    genre_size = dict.fromkeys(genres, 0)
    genre_lyrics = dict.fromkeys(genres)

    with open("./final.json") as f:
        data = json.load(f)["corpus"]
        for d in data:
            genre = d["label"]
            if genre_lyrics[genre] is None:
                genre_lyrics[genre] = []
            genre_lyrics[genre].append(d["data"])

    lyrics_words = []
    lyrics_embeddings = []
    labels = []
    num = 390
    for genre in genres:

        lyrics_words.extend(genre_lyrics[genre][:num])
        lyrics_embeddings.extend(extract_lyrics_embeddings(genre_lyrics[genre][:num]))
        sub_labels = [genre2int[genre]] * len(genre_lyrics[genre][:num])
        labels.extend(sub_labels)

        print("genre: {}, has {} lyrics, label length {}".format(
            genre, len(genre_lyrics[genre][:num]), len(sub_labels)))

    pickle.dump(lyrics_words, lyrics_words_file)
    pickle.dump(lyrics_embeddings, lyrics_embeddings_file)
    pickle.dump(labels, labels_file)


## extract lyrics as word embeddings
def extract_lyrics_embeddings(lyrics):
    """
    convert the lyrics into the sum of word embeddings
    divide the lyrics embedding by the length of the lyrics
    return a list of lyrics embeddings
    """
    lyrics_embeddings = []
    for words in lyrics:
        lyrics_embedding = 0        
        lyrics_length = len(words)
        for word in words:
            if word.lower() in word2vec_dict:
                 lyrics_embedding += np.array(word2vec_dict[word.lower()])
            else:
                 lyrics_embedding += 0.1 ## just an arbitrary number

        lyrics_embedding = lyrics_embedding/lyrics_length
        lyrics_embeddings.append(lyrics_embedding.tolist())
    return lyrics_embeddings


def naive_bayes_clf(lyrics_train, labels_train, lyrics_test, labels_test):
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

    text_clf = text_clf.fit(lyrics_train, labels_train)

    predicted = text_clf.predict(lyrics_train)
    print("NB, ACC on train:", np.mean(predicted == labels_train))
    predicted = text_clf.predict(lyrics_test)
    print("NB, ACC on test:", np.mean(predicted == labels_test))
    print("F1 micro:{}, F1 macro:{}, F1 weighted:{}".format(
        precision_recall_fscore_support(
            labels_test, predicted, average="micro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="macro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="weighted")[2]))
    # perform grid search for NB
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf__alpha': (1e-2, 1e-3),
    }
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5)
    gs_clf = gs_clf.fit(lyrics_train, labels_train)

    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    predicted = gs_clf.predict(lyrics_train)
    print("NB + gs, ACC on train:", np.mean(predicted == labels_train))
    predicted = gs_clf.predict(lyrics_test)
    print("NB + gs, ACC on test:", np.mean(predicted == labels_test))
    print("F1 micro:{}, F1 macro:{}, F1 weighted:{}".format(
        precision_recall_fscore_support(
            labels_test, predicted, average="micro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="macro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="weighted")[2]))


def svm_clf(lyrics_train, labels_train, lyrics_test, labels_test):
    text_clf_svm = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf-svm',
         SGDClassifier(
             loss='hinge', penalty='l2', alpha=1e-3, max_iter=100, tol=0.19,
             random_state=42)),
    ])

    text_clf_svm = text_clf_svm.fit(lyrics_train, labels_train)

    predicted = text_clf_svm.predict(lyrics_train)
    print("SVM, ACC on train:", np.mean(predicted == labels_train))
    predicted = text_clf_svm.predict(lyrics_test)
    print("SVM, ACC on test:", np.mean(predicted == labels_test))
    print("F1 micro:{}, F1 macro:{}, F1 weighted:{}".format(
        precision_recall_fscore_support(
            labels_test, predicted, average="micro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="macro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="weighted")[2]))
    # perform Grid Search for SVM
    parameters_svm = {
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'clf-svm__alpha': (1e-2, 1e-3),
    }
    gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1, cv=5)
    gs_clf_svm = gs_clf_svm.fit(lyrics_train, labels_train)
    print(gs_clf_svm.best_score_)
    print(gs_clf_svm.best_params_)

    predicted = text_clf_svm.predict(lyrics_train)
    print("SVM + gs, ACC on train:", np.mean(predicted == labels_train))
    predicted = text_clf_svm.predict(lyrics_test)
    print("SVM + gs, ACC on test:", np.mean(predicted == labels_test))
    print("F1 micro:{}, F1 macro:{}, F1 weighted:{}".format(
        precision_recall_fscore_support(
            labels_test, predicted, average="micro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="macro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="weighted")[2]))


def knn_clf(lyrics_train, labels_train, lyrics_test, labels_test):
    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(lyrics_train, labels_train)

    predicted = neigh.predict(lyrics_train)
    print ("KNN, ACC on train:", np.mean(predicted == labels_train))
    predicted = neigh.predict(lyrics_test)
    print ("KNN, ACC on test:", np.mean(predicted == labels_test))

    # perform grid search for KNN
    parameters = {'n_neighbors': [10, 25, 50, 100, 200, 300], 
                  'weights': ('uniform', 'distance')}
    gs_clf = GridSearchCV(neigh, parameters, cv=10, n_jobs=-1, scoring="accuracy")
    gs_clf = gs_clf.fit(lyrics_train, labels_train)

    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    predicted = gs_clf.predict(lyrics_train)
    print ("KNN + gs, ACC on train:", np.mean(predicted == labels_train))
    predicted = gs_clf.predict(lyrics_test)
    print ("KNN + gs, ACC on test:", np.mean(predicted == labels_test))

    print("F1 micro:{}, F1 macro:{}, F1 weighted:{}".format(
        precision_recall_fscore_support(
            labels_test, predicted, average="micro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="macro")[2],
        precision_recall_fscore_support(
            labels_test, predicted, average="weighted")[2]))



if __name__ == "__main__":
    ## load data for Naive Bayes, SVM, LSTM
    with open("test_lyrics_words", "wb") as lyrics_words_file, \
           open("test_lyrics_embeddings", "wb") as lyrics_embeddings_file, \
           open("test_labels", "wb") as labels_file:
        save_lyrics_label(lyrics_words_file, lyrics_embeddings_file, labels_file)
    ## load lyrics and labels lists from disk
    with open("test_lyrics_words", "rb") as lyrics_words_file, \
           open("test_lyrics_embeddings", "rb") as lyrics_embeddings_file, \
           open("test_labels", "rb") as labels_file:
        lyrics_words = pickle.load(lyrics_words_file)
        lyrics_embeddings = pickle.load(lyrics_embeddings_file)
        labels = pickle.load(labels_file)

    ## use the split function provided by sklearn to split STRATIFIED training and test set
    lyrics_words_train, lyrics_words_test, lyrics_embeddings_train, lyrics_embeddings_test, labels_train, labels_test = train_test_split(
        lyrics_words,
        lyrics_embeddings,
        labels,
        test_size=0.2,
        random_state=7,
        shuffle=True,
        stratify=labels)
    print("training set size:", len(lyrics_words_train))
    print("test set size:", len(lyrics_words_test))

    
    ## NB classifier
    print("\nNaive Bayes===============================")
    naive_bayes_clf(lyrics_words_train, labels_train, lyrics_words_test, labels_test)

    ## SVM classifier
    print("\nSVM=======================================")
    svm_clf(lyrics_words_train, labels_train, lyrics_words_test, labels_test)

    ## KNN classifier
    print("\nKNN=======================================")
    knn_clf(lyrics_embeddings_train, labels_train, lyrics_embeddings_test, labels_test)
    
    ## LSTM classifier
    # print("\nLSTM======================================")
    # LSTM.lstm_clf(lyrics_words, labels)

