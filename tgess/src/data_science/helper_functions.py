import yaml
import os
from pathlib import Path
import numpy as np
import pandas as pd
import random 
import string
# Get absolute path
mod_path = (Path(__file__).parent).resolve()
# mod_path = os.path.join(mod_path, "..")

def read_config(path=mod_path, config_file="default"):
    """
    Function to visualize correlation between features and target variable for both the in situ dataset and the simulation dataset.
    
    Args:
        config_file (str): Filepath of yaml configuration file

    Returns:
        config (dict): Parsed configuration file
    """

    # path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(path, "config", config_file+".yaml")

    # Load the configuration file
    stream = open(path, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    
    return config


def uniquify(path):
    """
    Function to save a file with a number suffix if the filename already exists.
    
    Args:
        path (str): Filepath of file

    Returns:
       path (str): Filepath with unique filename (with number suffix if appropriate)
    """

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_(" + str(counter) + ")" + extension
        counter += 1

    return path

def uniquify_folder(path):
    random.seed()
    path = path + str(''.join(random.SystemRandom().choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(10)))

    return path

def camel_case_split(s):
    words = [[s[0]]]
  
    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)
  
    return " ".join([(''.join(word)).capitalize() for word in words])

def flatten_time_series(df, feature_cols, target_col):
    """
    Flattens a dataset for use in a supervised model. Not suitable for recurrent models.
    New feature columns will have names with suffix for each timestep, e.g. lai_t-5 for LAI 5 weeks/months before last timestep
    Args:
        df (pd DataFrame):
        feature_cols (list of str): Feature column names
        target_col (str): column name of target variable

    return
        df (pd DataFrame): Flattened dataset
    """

    out_df = []

    for field_index in df["index"].unique():
        sub_df = df[df["index"]==field_index]

        n_timesteps = len(sub_df)
        cols = list(np.array([[col+"_t-{}".format(i) for col in feature_cols] for i in reversed(range(n_timesteps))]).flatten())
        ts_df = pd.DataFrame(sub_df[feature_cols].values.flatten()).T
        ts_df.columns = cols
        if type(target_col)==list:
            ts_df[target_col] = sub_df.iloc[0][target_col[0]]
        else:
            ts_df[target_col] = sub_df.iloc[0][target_col]
        out_df.append(ts_df)
    return pd.concat(out_df).interpolate()

def get_time_series_cols(df, cols, config):
    """
    Helper function to get column names for time series.
    """
    n_timesteps = config["experiment"]["data"]["n_timesteps"]

    print(list(df.columns))
    print(n_timesteps)
    n_timesteps = np.max(n_timesteps)

    out_cols = list(np.array([[col+"_t-{}".format(i) for col in cols] for i in reversed(range(n_timesteps))]).flatten())
    return out_cols

def flatten_list(lst):
    return [y for x in lst for y in x if type(x)==list]