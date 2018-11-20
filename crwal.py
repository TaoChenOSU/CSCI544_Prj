import json
from urllib.request import urlretrieve

def fetch_song_txt(artist):
	api_url = "https://itunes.apple.com/search?term={}+{}&media=music&entity=song&limit=200".format(*artist.split(" "))
	# print(api_url)
	saved_file = "./{}.txt".format(artist)
	urlretrieve(api_url, saved_file)
	return saved_file

def get_records_from_file(filename):
	records = []

	with open(filename, encoding="utf-8") as f:
		results = json.load(f)
		for result in results["results"]:
			record = dict()
			record["artist"] = result["artistName"]
			record["song_name"] = result["trackName"]
			record["genre"] = result["primaryGenreName"]
			records.append(record)

	return records

# saved_file = fetch_song_txt("Michael Jackson", 1)
# song_genres = get_song_genre(saved_file)
# print(song_genres)

def get_songs_of_artists(artists):
	records = []
	for artist in artists:
		saved_file = fetch_song_txt(artist)
		song_genres_i = get_records_from_file(saved_file)
		records.extend(song_genres_i)

	return records

if __name__ == '__main__':
	artists = ["Michael Jackson", "Kris Wu"]
	res = get_songs_of_artists(artists)
	for r in res:
		print(r)



