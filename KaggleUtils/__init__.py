"""Kaggle Utilities"""
import os
from zipfile import ZipFile


def download_extract_cleanup(kaggle_api, dataset, file_name, directory=None):
    """Download, extract, and cleanup Kaggle dataset

    Arguments:
        kaggle_api {KaggleApi} -- Instance of Kaggle API
        dataset {str} -- <owner>/<dataset> identifier
        file_name {str} -- The file to download

    Keyword Arguments:
        directory {str} -- The directory to place the downloaded file (default: {None})
    """

    kaggle_api.dataset_download_file(
        dataset, file_name, path=directory, quiet=False)
    extract_directory = directory if directory is not None else os.getcwd()
    extract_path = os.path.join(extract_directory, f"{file_name}.zip")
    ZipFile(extract_path).extractall(extract_directory)
    os.unlink(extract_path)