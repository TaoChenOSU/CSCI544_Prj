import ast
from bs4 import BeautifulSoup
import glob
import json
import requests


description_text = "description~~~"
headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'}

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

    json_dict = dict()
    json_dict["description"] = description_text
    json_dict["corpus"] = []
    for key, records in genre_records_map.items():
        with open(path + "test_{}.json".format(key), "w") as output:
            for record in records[:10]:    # for each label, only fetches the first 600 songs, to save time
                artist = record["artist"]
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


if __name__ == '__main__':
    records_dir = "./data/genre/"

    records = read_records_json(records_dir)
    # print(records)
    get_lyrics_of_songs(records)