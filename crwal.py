import glob
import json
import time
from urllib.request import urlretrieve


def load_artists(filename):
    with open(filename, "r", encoding="utf-8") as f:
        artists = [line.strip() for line in f]
    return artists


def fetch_songs_from_api(artists, output_dir):
    """
    Fetch a list of song records of artists from web and save to output_dir.

    Arg:
        artists: a list of artist names, e.g. "[Ariana Grande,]"
    """
    na = len(artists)
    i = 83

    while i < na:
        artist = artists[i]
        try:
            api_url = "https://itunes.apple.com/search?term={}&media=music&entity=song&limit=200".format(
                ("+").join(artist.split(" ")))
            response_data = output_dir + "{}.txt".format(artist)
            urlretrieve(api_url, response_data)

            print("Completed index {}, artist {}".format(i, artist))
            time.sleep(40)
            i += 1
        except Exception as e:
            print("Encounter error {} at index {}, artist {}".format(
                e, i, artist))
            time.sleep(200)


def extract_records(input_dir, output):
    """
    Extracts song records in the input_dir and output them to output.

    Arg:
        input_dir:
        output:
    """
    fo = open(output, "w", encoding="utf-8")

    for file in glob.glob(input_dir + "*"):
        with open(file, "r", encoding="utf-8") as fin:
            results = json.load(fin)
            artist_search_term = file.split("/")[-1].split(".")[0]
            for result in results["results"]:
                record = dict()
                record["artist"] = result["artistName"]
                record["song"] = result["trackName"]
                record["genre"] = result["primaryGenreName"]
                record["alternative_artist"] = artist_search_term
                fo.write("{}\n".format(record))

    fo.close()


# TODO:
# 1. Data cleansing: i.e. repeat data
# 2. Multiple genre labels.

if __name__ == '__main__':
    input_file = "./data/artnames_all_genres.txt"
    output_file = "./data/song_records.txt"
    records_dir = "./data/api/"

    # artists = load_artists(input_file)
    # records = fetch_songs_from_api(artists, records_dir)
    extract_records(records_dir, output_file)
