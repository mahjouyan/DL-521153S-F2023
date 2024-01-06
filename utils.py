import os
import re
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import DATASET, DATA_DIRECTORY


def stream_to_file(stream, file_name):
    block_size = 1024
    current_size = 0.0
    total_size = int(stream.headers["Content-Length"])
    progress_bar = tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        leave=False
        )

    with open(os.path.join(DATA_DIRECTORY, file_name), "wb") as out:
        for data in stream.iter_content(block_size):
            progress_bar.update(len(data))
            out.write(data)

    progress_bar.close()


def extract_tar(path):
    if not tarfile.is_tarfile(path):
        print("extract_tar: requested file is not a valid tar file!")

    with tarfile.open(path) as f:
        f.extractall(path=DATA_DIRECTORY)
        f.close()


def extract_zip(path):
    if not zipfile.is_zipfile(path):
        print("extract_zip: requested file is not a valid zip file!")

    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(DATA_DIRECTORY)
        f.close()


def extract_data(path):
    file_extention = Path(path).suffix

    if file_extention == ".tar":
        extract_tar(path)

    if file_extention == ".zip":
        extract_zip(path)

    pass


def download_from_google_drive(url, file_name):
    session = requests.Session()
    response = session.get(url, stream=True)

    # sometimes direct download works ?
    if not "text/html" in response.headers['content-type']:
        stream_to_file(response, file_name)
        return

    download_link = re.search(
        "action=\"(.*?)\"",
        response.text).group(1).replace(
        "&amp;",
        "&")

    response = session.post(download_link, stream=True)
    stream_to_file(response, file_name)


def download_from_url(url, file_name):
    session = requests.Session()
    response = session.get(url, stream=True)
    stream_to_file(response, file_name)


def download_dataset(url, file_name):
    print("[+] Downloading", file_name)
    # get full path name
    full_path = os.path.join(DATA_DIRECTORY, file_name)

    # download base on the dataset shared url
    if "docs.google.com" in url:
        download_from_google_drive(url, file_name)
    else:
        download_from_url(url, file_name)

    # extract downloaded data if needed
    print("[+] Extracting", full_path)
    extract_data(full_path)

    # remove compressed dataset
    os.remove(full_path)



def check_dataset_files():
    # make data directory if not exist
    if not os.path.exists(DATA_DIRECTORY):
        print("Data directory does not exist, creating...")
        os.mkdir(DATA_DIRECTORY)

    missing_dataset = dict()

    # check if required dataset exist
    for file_name, url in DATASET.items():
        if not os.path.exists(
            os.path.join(
                DATA_DIRECTORY,
                Path(file_name).stem)):
            missing_dataset.update({file_name: url})

    return missing_dataset

