import json
from urllib.request import urlretrieve


def fetch_songs_from_api(artist):
    """
    Fetch a list of song records of an artist from web. 

    Arg:
        artist, the name of the artist, e.g. "Ariana Grande"
    Return:
        a list of records: [{"artist":, "song_name":, "genre":, "alternate_artist"}]
    """

    api_url = "https://itunes.apple.com/search?term={}+{}&media=music&entity=song&limit=200".format(
        *artist.split(" "))
    # print(api_url)
    saved_file = "./data/{}.txt".format(artist)
    urlretrieve(api_url, saved_file)

    records = []
    with open(saved_file, "r", encoding="utf-8") as f:
        results = json.load(f)
        for result in results["results"]:
            record = dict()
            record["artist"] = result["artistName"]
            record["song_name"] = result["trackName"]
            record["genre"] = result["primaryGenreName"]
            record["alternate_artist"] = artist
            records.append(record)

    return records


def load_artists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        artists = [line.strip() for line in f]
    return artists


def get_songs_of_artists(artists):
    records = []
    for artist in artists:
        song_genres_i = fetch_songs_from_api(artist)
        records.extend(song_genres_i)

    return records


# TODO:
# 1. Data cleansing: i.e. repeat data
# 2. Multiple genre labels.

if __name__ == '__main__':
    artists = load_artists("./data/artists.txt")
    # print(artists)
    records = get_songs_of_artists(artists)
    # for r in records:
    #     print(r)
