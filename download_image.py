#git clone https://github.com/ultralytics/google-images-download

import bing_scraper
import os

def read_file_to_list(file_path):
    lines = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())
    return lines

file_path = 'name.txt'
name = read_file_to_list(file_path)

for i in range(len(name)):
    os.system("python bing_scraper.py --search '" + name[i] + "' --limit 200 --download")

