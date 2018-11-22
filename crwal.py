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
    i = 247

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


def extract_records(input_dir, output, summary):
    """
    Extracts song records in the input_dir and output them to output.

    Arg:
        input_dir:
        output:
        summary: file to contain total statistics of records.
    """
    files = glob.glob(input_dir + "*")
    genres = dict()
    records = []
    n_records_dedup = 0
    n_records = 0
    n_artists = len(files)
    n_true = 0
    total_brackets = 0

    for file in files:
        with open(file, "r", encoding="utf-8") as fin:
            results = json.load(fin)
            artist_search_term = file.split("/")[-1].split(".")[0]
            unoverlap_songs = set()
            songs_with_brackets = dict()

            for result in results["results"]:
                n_records += 1

                if not is_valid(result, unoverlap_songs):
                    continue

                song = result["trackName"]
                genre = result["primaryGenreName"]
                if "(" in song:
                    # save song idx to check if unbracketed
                    # songs in unoverlap_songs
                    songs_with_brackets[song] = n_records_dedup

                unoverlap_songs.add(song)
                genres[genre] = genres.get(genre, 0) + 1

                record = dict()
                record["artist"] = result["artistName"]
                record["song"] = song
                record["genre"] = genre
                record["alternative_artist"] = artist_search_term
                record["add_to_final_records"] = True
                records.append(record)
                n_records_dedup += 1

            n_true, total_brackets = filter_songs_with_brackets(
                songs_with_brackets, unoverlap_songs, records, n_true,
                total_brackets)

    save_records(records, genres, summary, output, n_records, n_records_dedup,
                 n_artists, n_true)

    return genres


def is_valid(record, unoverlap_songs):
    if "artistName" not in record:
        return False
    if "trackName" not in record:
        return False
    if "primaryGenreName" not in record:
        return False
    if record["trackName"] in unoverlap_songs:
        return False
    if "/" in record["primaryGenreName"]:
        return False
    return True


def save_records(records, genres, summary, output, n_records, n_records_dedup,
                 n_artists, n_true):
    n_added = 0
    fo = open(output, "w", encoding="utf-8")
    for record in records:
        if not record["add_to_final_records"]:
            continue
        fo.write("{}\n".format(record))
        n_added += 1
    fo.close()

    assert n_true == n_records_dedup - n_added

    fs = open(summary, "w", encoding="utf-8")
    fs.write("total: {} song records\n".format(n_records))
    fs.write(
        "total: {} song records after deduplication\n".format(n_records_dedup))
    fs.write("total: {} song records after remove deletable brackets\n".format(
        n_added))
    fs.write("total: {} artists\n".format(n_artists))
    fs.write("\n")
    for key, value in sorted(genres.items(), key=lambda x: -x[1]):
        fs.write("genre: {} has {} songs\n".format(key, value))
    fs.close()


def filter_songs_with_brackets(songs_with_brackets, unoverlap_songs, records,
                               has_true, total_brackets):
    for song, idx in songs_with_brackets.items():
        total_brackets += 1
        deduct_original_song = song.split("(")[0].strip()

        if deduct_original_song not in unoverlap_songs:
            continue
        has_true += 1
        records[idx]["add_to_final_records"] = False

    return has_true, total_brackets


# TODO:
# 1. Data cleansing: i.e. repeat data # deleted
# 2. Multiple genre labels. # deleted

if __name__ == '__main__':
    input_file = "./data/artnames_all_genres.txt"
    output_file = "./data/song_records.txt"
    summary_file = "./data/song_records_sumary.txt"
    records_dir = "./data/api/"

    # artists = load_artists(input_file)
    # fetch_songs_from_api(artists, records_dir)
    genres = extract_records(records_dir, output_file, summary_file)
