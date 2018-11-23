"""
This module contains multiple functions for fetching song records from
iTunes music api. The fetched data will be analysed further to extract
songs that belongs to eight genres for the classifier. The final records
are saved under folder data/genre/. 

To use the records to fetch lyrics from genius.com, you need to read 
records from files in data/genre/.
"""

import glob
import json
import time
from urllib.request import urlretrieve


def load_artists(filename):
    """Load artists list from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        artists = [line.strip() for line in f if line != ""]
    return artists


def fetch_songs_from_api(artists, output_dir):
    """
    Fetch a list of song records of artists from web and save to output_dir.

    Arg:
        artists: a list of artist names, e.g. "[Ariana Grande,]"
    """
    na = len(artists)
    i = 0

    while i < na:
        artist = artists[i]
        try:
            api_url = "https://itunes.apple.com/search?" + \
                      "term={}&media=music&entity=song&limit=200"
            api_url = api_url.format(("+").join(artist.split(" ")))
            response_data = output_dir + "{}.txt".format(artist)
            urlretrieve(api_url, response_data)

            print("Completed index {}, artist {}".format(i, artist))
            time.sleep(40)
            i += 1
        except Exception as e:
            print("Encounter error {} at index {}, artist {}".format(
                e, i, artist))
            i += 1
            time.sleep(200)


def extract_records(input_dir):
    """
    Extracts song records in the input_dir and output them to output.

    Arg:
        input_dir: a directory that contains all the songs fetched from 
            itunes.
    
    Return:
        records: a dictionary that contains all the records from all 
            artists, grouped by genre. i.e. {"Country": [record1, record2]}
        genres: a dictionary that contains the genres and the number of
            the songs belong to each genre.
        n_ori_records: the number of original records from the files in
            the input_dir.
        n_dedup_records: the number of deduplicated records.
        n_artists: the number of artists.
        n_has_unbracketed_song: the number of the songs that has its 
            unbrackted title in its artist songs. 
    """
    files = glob.glob(input_dir + "*")
    genres = dict()
    records = dict()
    n_dedup_records = 0
    n_ori_records = 0
    n_artists = len(files)
    n_has_unbracketed_song = 0
    total_brackets = 0

    for file in files:
        with open(file, "r", encoding="utf-8") as fin:
            results = json.load(fin)
            artist_search_term = file.split("/")[-1].split(".")[0]
            unoverlap_songs = set()
            songs_with_brackets = dict()

            for result in results["results"]:
                n_ori_records += 1

                if not is_valid(result, unoverlap_songs):
                    continue

                song = result["trackName"]
                genre = result["primaryGenreName"]

                unoverlap_songs.add(song)
                genres[genre] = genres.get(genre, 0) + 1

                record = dict()
                record["artist"] = result["artistName"]
                record["song"] = song
                record["genre"] = genre
                record["alternative_artist"] = artist_search_term
                record["add_to_final_records"] = True
                if genre not in records:
                    records[genre] = [record]
                else:
                    records[genre].append(record)

                if "(" in song:
                    # save song idx to check if unbracketed
                    # songs in unoverlap_songs
                    songs_with_brackets[(song,
                                         genre)] = len(records[genre]) - 1

                n_dedup_records += 1

            n_has_unbracketed_song, total_brackets = filter_songs_with_brackets(
                songs_with_brackets, unoverlap_songs, records,
                n_has_unbracketed_song, total_brackets)

    return (records, genres, n_ori_records, n_dedup_records, n_artists,
            n_has_unbracketed_song)


def is_valid(record, unoverlap_songs):
    """Check if the record can be added to records dict."""
    required_genres = [
        "Pop", "Rock", "Hip-Hop", "Blues", "Holiday", "Country", "Metal",
        "Children's Music"
    ]

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
    if record["primaryGenreName"] not in required_genres:
        return False
    return True


def save_records(records, genres, output, n_records_dedup, n_true):
    """Write records to output file, grouped by genre."""
    n_saved = 0

    for key in genres:
        genres[key] = 0

    for genre, recs in records.items():
        fo = open(output.format(genre), "w", encoding="utf-8")
        for record in recs:
            if not record["add_to_final_records"]:
                continue
            fo.write("{}\n".format(record))
            genres[record["genre"]] += 1
            n_saved += 1
        fo.close()

    assert n_true == n_records_dedup - n_saved
    return genres, n_saved


def save_data_dist(summary, genres, n_records, n_records_dedup, n_added,
                   n_artists):
    """Write the records distribution into a summary file."""

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
                               has_unbracketed_song, total_brackets):
    for song_genre, idx in songs_with_brackets.items():
        song, genre = song_genre[0], song_genre[1]

        total_brackets += 1
        deduct_original_song = song.split("(")[0].strip()

        if deduct_original_song not in unoverlap_songs:
            continue
        has_unbracketed_song += 1
        records[genre][idx]["add_to_final_records"] = False

    return has_unbracketed_song, total_brackets


if __name__ == "__main__":
    input_file = "./data/artnames_all_genres.txt"
    output_file = "./data/genre/{}.txt"
    summary_file = "./data/summary.txt"
    records_dir = "./data/api/"

    # fetch songs from api
    # artists = load_artists(input_file)
    # fetch_songs_from_api(artists, records_dir)

    # cleaning data and output them by genre
    (records, genres, n_ori_records, n_dedup_records, n_artists,
     n_has_unbracketed_song) = extract_records(records_dir)
    genres, n_saved = save_records(records, genres, output_file,
                                   n_dedup_records, n_has_unbracketed_song)
    save_data_dist(summary_file, genres, n_ori_records, n_dedup_records,
                   n_saved, n_artists)
