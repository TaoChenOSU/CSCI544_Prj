import ast
from bs4 import BeautifulSoup
import glob
import json
import os
import requests

description_text = "description~~~"
headers = {
    'user-agent':
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36'
}


def read_records_json(records_dir):
    """Read records from json files in records_dir."""
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


def is_english(s):
    """Check if a string contains only english chars."""
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_lyrics_of_songs(path,
                        genre_records_map,
                        valid_genres,
                        start_fetch_idx=0,
                        end_fetch_idx=600):
    """
    Fetch lyrics from genre_records_map.

    Filter the genres with valid_genres. i.e. If valid_genres = ["Country"],
    then only fetch lyrics of songs that belong to label Conuntry. 
    You can fetch a partition of the big song records, which may contain 
    thousands of songs. If you only wants to fetch 600 songs, set 
    start_fetch_idx to 0, end_fetch_idx to 600.
    If you want to fetch another 600, then set start_fetch_idx to 600,
    end_fetch_idx to 1200.
    Args:
        genre_records_map: the dictionary that contains genre and song 
            records, i.e. {genre1: [record_1, record_2]}
        valid_genres: a list that contains the genres to be crawled.
        start_fetch_idx: start indx of the fetch partition.
        end_fetch_idx: end idx of the fetch partition.
    """
    json_file = path + "test_{}.json"

    artists_songs = dict()

    for genre, records in genre_records_map.items():
        if genre not in valid_genres:
            continue

        json_dict = dict()
        json_dict["description"] = description_text
        json_dict["corpus"] = []
        with open(json_file.format(genre), "w") as output:
            # for each label, only fetches the first 600 songs, to save time
            for record in records[start_fetch_idx:end_fetch_idx]:
                artist = record["artist"]
                song_name = record["song"]

                genius_urls = get_possible_urls(record)
                get_lyrics = False

                # generate multiple urls to increase success rate
                for genius_url in genius_urls:
                    # print(genius_url, get_lyrics)
                    if get_lyrics:
                        break

                    req = requests.get(genius_url, headers=headers)

                    # make sure the page exists
                    if req.status_code == 200:
                        html = req.text
                        soup = BeautifulSoup(html, 'lxml')

                        lyrics_div = soup.find("div", {"class": "lyrics"})
                        if lyrics_div is not None:
                            if not is_english(lyrics_div.text):
                                continue

                            json_dict["corpus"].append({
                                "artist":
                                artist,
                                "song":
                                song_name,
                                "label":
                                record["genre"],
                                "data":
                                lyrics_div.text,
                            })
                            get_lyrics = True

            json_dict["number of corpus"] = len(json_dict["corpus"])

            json.dump(json_dict, output, indent=4)

        print("Finished genre {}, crawled {} lyrics".format(
            genre, json_dict["number of corpus"]))


def get_possible_urls(record):
    """Generate all possible urls."""
    possible_urls = []

    artist = record["artist"]
    song_name = record["song"]
    alternative_artist = record["alternative_artist"]
    genre = record["genre"]

    artist = artist.replace(" ", "-")
    artist = artist.replace("&", "and")
    artist = artist.replace(",", "and")
    song_name = song_name.replace(" ", "-")
    alternative_artist = alternative_artist.replace(" ", "-")
    genre = genre.replace(
        " ", "-") if genre != "Children's Music" else "Children-songs"

    artists_search_terms = [artist, alternative_artist, genre]

    for st in artists_search_terms:
        possible_urls.append("https://genius.com/{}-{}-lyrics".format(
            st, song_name))

    return possible_urls


if __name__ == '__main__':
    records_dir = "./data/genre/"
    records = read_records_json(records_dir)
    path = "./data/lyrics_test/"  # change the name of this!!!!!
    if not os.path.exists(path):
        os.mkdir(path)

    valid_genres = [
        "Country",
    ]
    # please set a reasonable start and end idx here
    get_lyrics_of_songs(
        path,
        records,
        valid_genres,
        start_fetch_idx=11500,
        end_fetch_idx=20000)
