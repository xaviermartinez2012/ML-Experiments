"""ML-Experiments"""
import os
import pandas
from zipfile import ZipFile


class experiment:
    def __init__(self, kaggle_api, dataset, dataset_target,
                 download_directory):
        """Experiment encapsulates a ML experiment

        Arguments:
            kaggle_api {KaggleApi} -- Instance of KaggleApi
            dataset {str} -- <owner/resource>
            dataset_target {str} -- <filename>.<ext>
            download_directory {str} -- Path to place the downloaded dataset (relative to $VIRTUAL_ENV)
        """

        self.kaggle_api = kaggle_api
        self.dataset = dataset
        self.dataset_target = dataset_target
        self.download_directory = os.path.join(os.environ['VIRTUAL_ENV'],
                                               download_directory)
        self.dataset_file = os.path.join(download_directory, dataset_target)
        self.dataframe = self.initialize_dataframe()

    def initialize_dataframe(self):
        """Initialize a DataFrame from a Kaggle dataset"""

        if not os.path.exists(self.dataset_file):
            self.kaggle_api.dataset_download_file(
                self.dataset,
                self.dataset_target,
                path=self.download_directory)
            extract_target = f"{self.dataset_file}.zip"
            ZipFile(extract_target).extractall(self.download_directory)
            os.unlink(extract_target)

        return pandas.read_csv(self.dataset_file)

    def describe_dataset(self):
        """Describe the DataFrame"""

        format_str = "{:<11}: {}"
        print(format_str.format('Dataset', self.dataset))
        print(format_str.format('File', self.dataset_target))
        print(format_str.format('Attributes', ', '.join(list(self.dataframe))))
        dataset_shape = "{} x {}".format(*self.dataframe.shape)
        print(format_str.format('Shape', dataset_shape))


def env_sanity_check():
    """Assert that the environment is configured correctly"""

    environment_variables = os.environ.keys()
    required_variables = ["VIRTUAL_ENV", "KAGGLE_USERNAME", "KAGGLE_KEY"]
    assert all([
        variable in environment_variables for variable in required_variables
    ]), "virtual environment is not properly configured"
