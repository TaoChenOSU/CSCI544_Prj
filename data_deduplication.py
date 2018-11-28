from difflib import SequenceMatcher
import json
import glob


def similarity(a, b):
    """Return the similarity of string a and string b."""
    return SequenceMatcher(None, a, b).ratio()


def load_json_from_file(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_to_file(file, json_dict):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=2)


def deduplication_by_song(input_file, output_file, similarity_threshold=0.9):
    json_dict = load_json_from_file(input_file)

    song_data = dict()
    corpus = json_dict["corpus"]
    new_corpus = []

    for c in corpus:
        song = c["song"]
        genre = c["label"]
        data = c["data"]

        if song not in song_data:
            song_data[song] = [data]
            new_corpus.append({"label": genre, "data": data})
        else:
            is_duplicated = False
            for data_in_song in song_data[song]:
                if similarity(data, data_in_song) > similarity_threshold:
                    is_duplicated = True
                    print("============================================")
                    print("Duplication detected in song {}".format(song))
                    print("----Already added----")
                    print(data_in_song)
                    print("----Current----")
                    print(data)
                    break
            if not is_duplicated:
                song_data[song].append(data)
                new_corpus.append({"label": genre, "data": data})

    json_dict["corpus"] = new_corpus
    json_dict["number of corpus"] = len(new_corpus)
    print("Before deduplication, {} songs".format(len(corpus)))
    print("After deduplication, {} songs".format(len(new_corpus)))
    save_json_to_file(output_file, json_dict)


def deduplication_by_data(input_file,
                          output_file,
                          similarity_threshold=0.8,
                          data_length_threshold=200):
    json_dict = load_json_from_file(input_file)

    datas = set()
    corpus = [c["data"] for c in json_dict["corpus"]]
    new_corpus = []

    for c in corpus:
        genre = c["label"]
        data = c["data"]

        if len(data) < data_length_threshold:
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print("data too short")
            print(data)
            continue

        if data not in datas:
            datas.add(data)
            new_corpus.append({"label": genre, "data": data})

        is_duplicated = False
        for d in datas:
            sim = similarity(data, d)
            if sim > similarity_threshold:
                is_duplicated = True
                print("============================================")
                print("Duplication detected at", sim)
                print("----Current----")
                print(data)
                print("---------------")
                print("----Already added----")
                print(d)
                break
        if not is_duplicated:
            datas.append(data)
            new_corpus.append({"label": genre, "data": data})

    json_dict["corpus"] = new_corpus
    json_dict["number of corpus"] = len(new_corpus)
    print("Before deduplication, {} songs".format(len(corpus)))
    print("After deduplication, {} songs".format(len(new_corpus)))
    save_json_to_file(output_file, json_dict)


def generate_final_json():
    final = "./final.json"
    files = glob.glob("./data/lyrics/*")

    json_dict = dict()
    json_dict["description"] = ""
    json_dict["corpus"] = []
    for f in files:
        corpus = load_json_from_file(f)["corpus"]
        json_dict["corpus"].extend(corpus)

    json_dict["authors"] = dict({
        "author1": "Tao Chen",
        "author2": "Mu Yang",
        "author3": "Chang Su",
        "author4": "Zhe Yang"
    })
    json_dict["emails"] = dict({
        "email1": "taochen@usc.edu",
        "email2": "yangmu@usc.edu",
        "email3": "csu272@usc.edu",
        "email4": "zheyang@usc.edu"
    })
    save_json_to_file(final, json_dict)


if __name__ == "__main__":
    generate_final_json()
    # genres = ["Rock"]
    # for genre in genres:
    #     input_file = "./data/lyrics_dedup_by_data/{}.json".format(genre)
    #     output_file = "./data/lyrics_dedup_by_data/{}2.json".format(genre)
    #     print("Processing genre {}".format(genre))
    #     deduplication_by_data(input_file, output_file)
