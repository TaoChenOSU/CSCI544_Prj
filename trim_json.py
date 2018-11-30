"""
To use this program, changte the INPUT_DIRECTORY annd OUTPUT_DIRECTORY
in the main method to the directory where current json file are stored
and the directory where you want to store those json file aftre trim.

"""
import json
import os
import re


def trim_json(input_directory, file_name, output_directory):
    """Remove pattern [] in file_name."""
    in_filepath = input_directory + file_name
    out_filepath = output_directory + file_name

    with open(in_filepath, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    for song in data_dict["corpus"]:
        # replace '[XXX 1]' with ''
        song["data"] = re.sub(r'\[.*\]', '', song["data"])

    if not os.path.exists(out_filepath):
        with open(out_filepath, "x", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)
    else:
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)


if __name__ == "__main__":
    # the directory where target Json are; YOU MAY WANT TO CHANGE THIS!
    INPUT_DIRECTORY = "./data/lyrics/"
    # where the output Json files are stored
    OUTPUT_DIRECTORY = "./trim_result/"

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    try:
        files = os.listdir(INPUT_DIRECTORY)
        for file in files:
            trim_json(INPUT_DIRECTORY, file, OUTPUT_DIRECTORY)
    except FileNotFoundError:
        print("INPUT_DIRECTORY does not exist")
