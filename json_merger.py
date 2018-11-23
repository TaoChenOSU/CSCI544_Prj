import glob
import json
import os


class JsonMerger:
    def __init__(self):
        pass

    def merge_two_file(self, file1, file2, new_file=None, dedup=False):
        lyrics = self.load_json_from_file(file1)
        lyrics2 = self.load_json_from_file(file2)
        lyrics["corpus"].extend(lyrics2["corpus"])
        lyrics["number of corpus"] += lyrics2["number of corpus"]
        print("Finished genre {}, total {} lyrics".format(
            lyrics["corpus"][0]["label"], lyrics["number of corpus"]))

        if dedup:
            lyrics["corpus"] = self.dedup_lyrics(lyrics["corpus"])

        new_file = file1 if new_file is None else new_file
        self.save_json_to_file(new_file, lyrics)

    def dedup_lyrics(self, corpus):
        songs = dict()
        new_corpus = []

        for c in corpus:
            artist = c["artist"]
            song = c["song"]
            if (artist, song) not in songs:
                songs[(artist, song)] = True
                new_corpus.append(c)

        return new_corpus

    def merge_two_folder(self, folder1, folder2, new_folder=None):
        files1 = glob.glob(os.path.join(folder1, "*"))
        files2 = glob.glob(os.path.join(folder2, "*"))

        new_folder = folder1 if new_folder is None else new_folder
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)

        for file1 in files1:
            file_name = file1.split("/")[-1]
            file2 = os.path.join(folder2, file_name)
            assert file2 in files2
            self.merge_two_file(file1, file2,
                                os.path.join(new_folder, file_name))

    def load_json_from_file(self, file):
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json_to_file(self, file, json_dict):
        with open(file, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)


if __name__ == "__main__":
    json_merger = JsonMerger()
    json_merger.merge_two_folder("./lyrics1", "./lyrics2", "./lyrics3")
