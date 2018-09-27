"""
    Performs Linear Regression on Kaggle Dataset
    Title: Google Play Store Apps
    URL: https://www.kaggle.com/lava18/google-play-store-apps
    File: googleplaystore.csv
"""
import pandas
import os
import KaggleUtils
from kaggle import api as kaggle_api


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
        KaggleUtils.download_extract_cleanup(kaggle_api,
                                             "lava18/google-play-store-apps",
                                             file_name, dataset_directory)
    # Create DataFrame & describe dataset
    df = pandas.read_csv(dataset_file)
    print(f"{'File':<11}: {file_name}")
    print(f"{'Attributes':<11}: {', '.join(list(df))}")
    print("{0:<11}: {1} x {2}".format('Shape', *df.shape))


if __name__ == '__main__':
    main()