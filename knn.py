import json
import numpy as np 
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


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

def extract_lyrics_from_file(f, num=None):
    """
    extract lyrics data from one json file.
    when num is not None, return num number of lyrics, otherwise return all lyrics in the file
    convert the lyrics into the sum of word embeddings
    divide the lyrics embedding by the length of the lyrics
    return a list of lyrics
    """
    lyrics = []
    data = json.load(f)

    if not num:
        num = len(data["corpus"])

    for i in range(num):
        lyrics_embedding = 0
        lyrics_length = len(data["corpus"][i]["data"])
        for word in data["corpus"][i]["data"]:
            if word.lower() in word2vec_dict:
                 lyrics_embedding += np.array(word2vec_dict[word.lower()])
            else:
                 lyrics_embedding += 0.1

        lyrics_embedding = lyrics_embedding/lyrics_length
        lyrics.append(lyrics_embedding.tolist())
    return lyrics

def save_lyrics_label(lyrics_file, labels_file):
    """
    Save lyrics and labels as two lists. Then use pickle to save them to disk
    """

    genres = ["Hip-Hop", "Rock", "Pop", "Country", "Metal", "Holiday", "Children's Music", "Blues"]
    genre2int = {"Hip-Hop":1, 
        "Rock":2, 
        "Pop":3, 
        "Country":4, 
        "Metal":5, 
        "Holiday":6, 
        "Children's Music":7, 
        "Blues":8
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
    with open("testNB_lyrics", "wb") as lyrics_file, open("testNB_labels", "wb") as labels_file:
        save_lyrics_label(lyrics_file, labels_file)
    # load lyrics and labels lists from disk
    with open("testNB_lyrics", "rb") as lyrics_file, open("testNB_labels", "rb") as labels_file:
        lyrics = pickle.load(lyrics_file)
        labels = pickle.load(labels_file)

    # split training and test set.
    # last 300 used as test, others used as training
    lyrics_shuffle, labels_shuffle = shuffle(lyrics, labels, random_state=0)
    lyrics_train = lyrics_shuffle[:-300]
    labels_train = labels_shuffle[:-300]
    lyrics_test = lyrics_shuffle[-300:]
    labels_test = labels_shuffle[-300:]
    print ("training set size:", len(lyrics_train))
    print ("test set size:", len(lyrics_test))


    # KNN classifier
    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(lyrics_train, labels_train)

    predicted = neigh.predict(lyrics_train)
    print ("NB, ACC on train:", np.mean(predicted == labels_train))
    predicted = neigh.predict(lyrics_test)
    print ("NB, ACC on test:", np.mean(predicted == labels_test))

    # perform grid search for KNN
    parameters = {'n_neighbors': [10, 50, 100, 200, 300]}
    gs_clf = GridSearchCV(neigh, parameters, cv=10, n_jobs=-1, scoring="accuracy")
    gs_clf = gs_clf.fit(lyrics_train, labels_train)

    print(gs_clf.best_score_)
    print(gs_clf.best_params_)

    predicted = gs_clf.predict(lyrics_train)
    print ("KNN + gs, ACC on train:", np.mean(predicted == labels_train))
    predicted = gs_clf.predict(lyrics_test)
    print ("KNN + gs, ACC on test:", np.mean(predicted == labels_test))