import ast
from bs4 import BeautifulSoup
import glob
import json
import requests


description_text = "description~~~"

def read_records_json(records_dir):
      genre_records_map = dict()
      files = glob.glob(records_dir + "*")
      for file in files:
             # your current line
             genre = file.split("/")[-1].split(".")[0]
             genre_records_map[genre] = []
             with open(file, "r") as fp:
                    for line in fp:
                         genre_records_map[genre].append(ast.literal_eval(line))
      return genre_records_map

def get_lyrics_of_songs(genre_records_map):

    path = "./data/lyrics/"

    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}
    artists_songs = dict()

    for key, records in genre_records_map.items():
        json_dict = dict()
        json_dict["description"] = description_text
        json_dict["corpus"] = []
        with open(path + "test_{}.json".format(key), "w") as output:
            for record in records[:600]:    # for each label, only fetches the first 600 songs, to save time
                artist = record["artist"]
                if artist == "Kris Wu":
                    continue

                song_name = record["song"]

                # replace spaces with hyphens
                artist = artist.replace(" ", "-")
                song_name = song_name.replace(" ", "-")

                genius_url = "https://genius.com/{}-{}-lyrics".format(artist, song_name)
                # print(genius_url)

                # using the python requests library because it's easy to use
                req = requests.get(genius_url, headers=headers)

                # make sure the page exists
                if req.status_code == 200:
                    html = req.text
                    soup = BeautifulSoup(html, 'lxml')

                    lyrics_div = soup.find("div", {"class": "lyrics"})
                    if lyrics_div is not None:
                        # print(lyrics_div.text)
                        json_dict["corpus"].append({
                            "artist": artist,
                            "song": song_name,
                            "label": record["genre"],
                            "data": lyrics_div.text,
                        })

            json_dict["authors"] = dict({
                "author1": "Tao Chen",
                "author2": "Yang Mu",
                "author3": "Su Chang",
                "author4": "Yang Zhe"
            })
            json_dict["emails"] = dict({
                "email1": "taochen@usc.edu",
                "email2": "yangmu@usc.edu",
                "email3": "csu272@usc.edu",
                "email4": "zheyang@usc.edu"
            })

            json.dump(json_dict, output, indent=4)


# TODO:
# 1. Data cleansing: i.e. repeat data # deleted
# 2. Multiple genre labels. # deleted

if __name__ == '__main__':
    records_dir = "./data/genre/"

    # artists = load_artists(input_file)
    # fetch_songs_from_api(artists, records_dir)
    # genres = extract_records(records_dir, output_file, summary_file)
    records = read_records_json(records_dir)
    # print(records)
    get_lyrics_of_songs(records)