#! /usr/bin/env python3
"""
    Performs Linear Regression on Kaggle Dataset
    Title: Google Play Store Apps
    URL: https://www.kaggle.com/lava18/google-play-store-apps
    File: googleplaystore.csv
"""
from kaggle import api as kaggle_api
from ml_experiments import experiment, env_sanity_check


def main():
    """Main function"""
    # env sanity check
    env_sanity_check()
    # Set file paths
    dataset = "lava18/google-play-store-apps"
    dataset_target = "googleplaystore.csv"
    download_directory = "Experiments/Regression/datasets"
    # initialize experiment
    ml_experiment = experiment(kaggle_api, dataset, dataset_target,
                               download_directory)
    ml_experiment.describe_dataset()


if __name__ == '__main__':
    main()