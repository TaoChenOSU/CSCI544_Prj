import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

import json


def extract_lyrics_from_file(f, num=None):
    """
    Extract lyrics data from one json file.

    When num is not None, return num number of lyrics, otherwise return all 
    lyrics in the file.

    Return:
        A list of lyrics.
    """
    data = json.load(f)
    lyrics = [c["data"] for c in data["corpus"]]
    return lyrics[:num]


def save_lyrics_label(lyrics_file, labels_file):
    """
    Save lyrics and labels as two lists. 

    Then use pickle to save them to disk.
    """

    genres = [
        "Hip-Hop", "Rock", "Pop", "Country", "Metal", "Holiday",
        "Children's Music", "Blues"
    ]
    genre2int = {
        "Hip-Hop": 1,
        "Rock": 2,
        "Pop": 3,
        "Country": 4,
        "Metal": 5,
        "Holiday": 6,
        "Children's Music": 7,
        "Blues": 8
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

    lyrics = []
    labels = []
    num = 390
    for genre in genres:
        sub_lyrics = genre_lyrics[genre][:num]
        sub_labels = [genre2int[genre]] * len(sub_lyrics)
        lyrics.extend(sub_lyrics)
        labels.extend(sub_labels)
        print("genre: {}, has {} lyrics, label length {}".format(
            genre, len(sub_lyrics), len(sub_labels)))

    pickle.dump(lyrics, lyrics_file)
    pickle.dump(labels, labels_file)


if __name__ == "__main__":
    # save lyrics and labels lists to disk
    with open("testNB_lyrics", "wb") as lyrics_file, open(
            "testNB_labels", "wb") as labels_file:
        save_lyrics_label(lyrics_file, labels_file)
    # load lyrics and labels lists from disk
    with open("testNB_lyrics", "rb") as lyrics_file, open(
            "testNB_labels", "rb") as labels_file:
        lyrics = pickle.load(lyrics_file)
        labels = pickle.load(labels_file)

    # split training and test set.
    # last 300 used as test, others used as training
    lyrics_shuffle, labels_shuffle = shuffle(lyrics, labels, random_state=0)
    split_idx = -len(lyrics_shuffle) // 10 * 2
    lyrics_train = lyrics_shuffle[:split_idx]
    labels_train = labels_shuffle[:split_idx]
    lyrics_test = lyrics_shuffle[split_idx:]
    labels_test = labels_shuffle[split_idx:]
    print("training set size:", len(lyrics_train))
    print("test set size:", len(lyrics_test))

    # NB classifier
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

    # perform grid search for NB
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
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

    # SVM classifier
    text_clf_svm = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf-svm',
         SGDClassifier(
             loss='hinge',
             penalty='l2',
             alpha=1e-3,
             max_iter=10,
             random_state=42)),
    ])

    text_clf_svm = text_clf_svm.fit(lyrics_train, labels_train)

    predicted = text_clf_svm.predict(lyrics_train)
    print("SVM, ACC on train:", np.mean(predicted == labels_train))
    predicted = text_clf_svm.predict(lyrics_test)
    print("SVM, ACC on test:", np.mean(predicted == labels_test))

    # perform Grid Search for SVM
    parameters_svm = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)],
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
