#! /usr/bin/env python3
"""
    Performs Linear Regression on Kaggle Dataset
    Title: Google Play Store Apps
    URL: https://www.kaggle.com/lava18/google-play-store-apps
    File: googleplaystore.csv
"""
import re
import numpy as np
from kaggle import api as kaggle_api
from ml_experiments import experiment, env_sanity_check
from pandas import to_datetime


def main():
    """Main function"""
    # env sanity check
    env_sanity_check()
    # initialize experiment
    dataset = "lava18/google-play-store-apps"
    dataset_target = "googleplaystore.csv"
    download_directory = "Experiments/Regression/datasets"
    ml = experiment(kaggle_api, dataset, dataset_target, download_directory)
    # data cleaning
    numeric_conversions = []
    ## drop duplicate apps
    stripped_apps = ml.df.App.apply(lambda x: " ".join((x.strip()).split()))
    ml.reassign_attribute("App", stripped_apps)
    ml.df.drop_duplicates(subset="App", inplace=True)
    ## - Installs :
    ##  Replace "+"/"," -> "" and convert number of installs to integer
    cleaned_installs = ml.df.Installs.apply(lambda x: re.sub(r"\+|,", "", x))
    ml.reassign_attribute("Installs", cleaned_installs)
    ##  Remove convert installs == "Free" to NaN
    ml.reassign_attribute("Installs", ml.df.Installs.replace("Free", np.NaN))
    numeric_conversions.append("Installs")
    ## - Size :
    ##  source: https://www.kaggle.com/sabasiddiqi/google-play-store-apps-data-cleaning
    ml.reassign_attribute("Size", ml.df.Size.str.replace("k", "e+3"))
    ml.reassign_attribute("Size", ml.df.Size.str.replace("M", "e+6"))
    ml.reassign_attribute("Size",
                          ml.df.Size.replace("Varies with device", np.nan))
    ml.reassign_attribute("Size", ml.df.Size.replace("1,000+", 1000))
    numeric_conversions.append("Size")
    ## - Reviews
    ##  source: https://www.kaggle.com/sabasiddiqi/google-play-store-apps-data-cleaning
    ml.df.drop(ml.identitify_non_numeric("Reviews").index, inplace=True)
    ## - Price
    ##  source: https://www.kaggle.com/sabasiddiqi/google-play-store-apps-data-cleaning
    stripped_prices = ml.df.Price.apply(lambda x: x.strip("$"))
    ml.reassign_attribute("Price", stripped_prices)
    numeric_conversions.append("Price")
    ## - Genre
    ##  source: https://www.kaggle.com/sabasiddiqi/google-play-store-apps-data-cleaning
    primary_genre = ml.df.Genres.apply(lambda x: x.split(";")[0]).values
    ml.df["PrimaryGenre"] = primary_genre
    secondary_genre = ml.df.Genres.apply(lambda x: x.split(";")[-1]).values
    ml.df["SecondaryGenre"] = secondary_genre
    ml.df.drop(columns=["Genres"], inplace=True)
    ## - Last Updated
    ml.reassign_attribute("Last Updated", to_datetime(ml.df["Last Updated"]))
    ## - Drop Current Ver, Android Ver
    ml.df.drop(columns=["Current Ver", "Android Ver"], inplace=True)
    for attribute in numeric_conversions:
        ml.convert_to_numeric(attribute)
    ## drop NaNs
    ml.df.dropna(inplace=True)
    ml.describe_dataset()


if __name__ == "__main__":
    main()