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
        self.download_directory = os.path.join(os.environ["VIRTUAL_ENV"],
                                               download_directory)
        self.dataset_file = os.path.join(self.download_directory,
                                         dataset_target)
        self.df = self.initialize_dataframe()

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

    def reassign_attribute(self, attribute, series):
        """Reassigns column in dataset using best practices

        Arguments:
            attribute {str} -- The attribute to reassign
            series {Series} -- The Series to use for reassignment
        """

        self.df.loc[:, attribute] = series.values

    def identitify_non_numeric(self, attribute):
        """Identifies non-numeric values in given an attribute in the dataset

        Arguments:
            attribute {str} -- The attribute to investigate

        Returns:
            DataFrame -- Non-numeric DataFrame
        """

        not_numeric = self.df[~self.df[attribute].str.isnumeric()]
        return not_numeric

    def convert_to_numeric(self, attribute):
        """Convert data associated to attribute to numeric

        Arguments:
            attribute {str} -- The attribute to target
        """

        self.reassign_attribute(attribute,
                                pandas.to_numeric(self.df[attribute]))

    def describe_dataset(self):
        """Describe the DataFrame"""

        format_str = "{:<11}: {}"
        print(format_str.format('Dataset', self.dataset))
        print(format_str.format('File', self.dataset_target))
        print(format_str.format('Attributes', ', '.join(list(self.df))))
        dataset_shape = "{} x {}".format(*self.df.shape)
        print(format_str.format('Shape', dataset_shape))
        print(self.df.sample(5))

    def describe_non_numeric(self, attribute):
        """Describes the non-numeric data in the dataset given an attribute

        Arguments:
            attribute {str} -- The attribute to investigate
        """

        format_str = "{:<20}: {}"
        not_numeric = self.identitify_non_numeric(attribute)
        unique_not_numeric = not_numeric.unique()
        print(format_str.format('Total non-numeric', not_numeric.sum()))
        print(
            format_str.format('Total unique non-numeric',
                              unique_not_numeric.sum()))
        print(f"{'Unique non-numeric':<20}:\n{unique_not_numeric}")


def env_sanity_check():
    """Assert that the environment is configured correctly"""

    environment_variables = os.environ.keys()
    required_variables = ["VIRTUAL_ENV", "KAGGLE_USERNAME", "KAGGLE_KEY"]
    requirements = [
        variable in environment_variables for variable in required_variables
    ]
    assert all(requirements), "virtual environment is not properly configured"
