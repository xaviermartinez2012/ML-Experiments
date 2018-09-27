"""
    Performs Linear Regression on Kaggle Dataset
    Title: Google Play Store Apps
    URL: https://www.kaggle.com/lava18/google-play-store-apps
    File: googleplaystore.csv
"""
import pandas
import os
from kaggle import api as kaggle_api
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


def main():
    """Main function"""
    # venv sanity check
    environment_variables = os.environ.keys()
    required_variables = ["VIRTUAL_ENV", "KAGGLE_USERNAME", "KAGGLE_KEY"]
    assert all([
        variable in environment_variables for variable in required_variables
    ]), "virtual environment is not properly configured"

    # Set path
    dataset_directory = f"{os.environ['VIRTUAL_ENV']}/Experiments/Regression/datasets"
    file_name = "googleplaystore.csv"
    dataset_file = os.path.join(dataset_directory, file_name)
    # Load or download dataset
    if not os.path.exists(dataset_file):
        download_extract_cleanup(kaggle_api, "lava18/google-play-store-apps",
                                 file_name, dataset_directory)
    # Create DataFrame & describe dataset
    df = pandas.read_csv(dataset_file)
    print(f"{'File':<11}: {file_name}")
    print(f"{'Attributes':<11}: {', '.join(list(df))}")
    print("{0:<11}: {1} x {2}".format('Shape', *df.shape))


if __name__ == '__main__':
    main()