# NOTE: This script produces the following files:
#       - movie_data.csv.gz
#       - movie_data.csv


import gzip
import shutil
import requests

if __name__ == '__main__':
    # URL pointing to the compressed CSV file on GitHub
    url = ("https://github.com/rasbt/machine-learning-book/raw/main/ch08/movie_data.csv.gz")

    # Grab the last element in the split of "url" which is "movie_data.csv.gz"
    filename = url.split("/")[-1]

    # Open "movie_data.csv.gz" in binary mode, acquire the resource at the url, and save it
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    # Open the compressed file in read binary model
    with gzip.open('movie_data.csv.gz', 'rb') as f_in:
        # Then open the csv file also write binary mode
        with open('movie_data.csv', 'wb') as f_out:
            # copies the content of the compressed file into the csv file
            shutil.copyfileobj(f_in, f_out)