import json
import re
import os

"""
To use this program, changte the INPUT_DIRECTORY annd OUTPUT_DIRECTORY
in the main method to the directory where current json file are stored
and the directory where you want to store those json file aftre trim.

"""

def trim_json(input_directory, file_name, output_directory):
    """
    remove pattern []
    """
    in_filepath = input_directory + file_name
    out_filepath = output_directory + file_name

    # open file from imput_directory
    with open(in_filepath, "r", encoding="utf-8") as f:
        data_dict = json.load(f)
    
    for song in data_dict["corpus"]:  # replace '[XXX 1]' with ''
            song["data"] = re.sub(r'\[.*\]','' , song["data"]) 
            #song["data"] = re.sub(r'\n','' , song["data"]) 

    # write to a new file in output_directory
    if not os.path.exists(out_filepath): # file not exists, create and write mode
        with open(out_filepath, "x", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)
    else: # file exists, write mode
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(data_dict, f, indent=4)

        

if __name__ == "__main__":
    INPUT_DIRECTORY = "./todo/"  # the directory where target Json are; YOU MAY WANT TO CHANGE THIS!
    OUTPUT_DIRECTORY = "./trim_result/"  # where the output Json files are stored

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)
    
    try:
        files = os.listdir(INPUT_DIRECTORY)
        for file in files:
            trim_json(INPUT_DIRECTORY, file, OUTPUT_DIRECTORY)
            
    except FileNotFoundError:
        print("INPUT_DIRECTORY does not exist")

    
