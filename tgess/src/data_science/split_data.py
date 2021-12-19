import pandas as pd 
import numpy as np 
import os
from .helper_functions import *
import random

from sklearn.model_selection import train_test_split

# Get root folder of project
root_path = (Path(__file__).parent / ".." / ".." ).resolve()

def split_in_situ_data(config, random_seed=1):

    # Read variables from config
    filename = config["experiment"]["data"]["in_situ_dataset"]
    path = config["experiment"]["data"]["path"]
    train_samples = config["experiment"]["split"]["train_samples"]
    target_col = config["experiment"]["data"]["in_situ_target_col"]

    in_situ_df = pd.read_csv(os.path.join(root_path, path, filename))

    holdout_size = len(in_situ_df) - train_samples

    X = in_situ_df.drop(columns=[target_col])
    y = in_situ_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=holdout_size, shuffle=True, random_state=random_seed)

    train_set = pd.DataFrame(X_train).join(pd.DataFrame(y_train))
    holdout_set = pd.DataFrame(X_test).join(pd.DataFrame(y_test))

    # print("Saving in situ training set to {}".format(os.path.join(root_path, path, filename[:-4]+"_{}_{}_train.csv".format(train_samples, random_seed))))
    # train_set.to_csv(os.path.join(root_path, path, filename[:-4]+"_{}_{}_train.csv".format(train_samples, random_seed)))

    # print("Saving in situ holdout set to {}".format(os.path.join(root_path, path, filename[:-4]+"_{}_{}_holdout.csv".format(train_samples, random_seed))))
    # holdout_set.to_csv(os.path.join(root_path, path, filename[:-4]+"_{}_{}_holdout.csv".format(train_samples, random_seed)))

    return train_set, holdout_set

def split_in_situ_time_series_data(config, random_seed=1, index_col="index"):
    # Read variables from config
    filename = config["experiment"]["data"]["in_situ_dataset"]
    path = config["experiment"]["data"]["path"]
    train_samples = config["experiment"]["split"]["train_samples"]
    target_col = config["experiment"]["data"]["in_situ_target_col"]

    in_situ_df = pd.read_csv(os.path.join(root_path, path, filename))#.reset_index()

    holdout_size = len(in_situ_df[index_col].unique()) - train_samples

    X = in_situ_df.drop(columns=[target_col])
    y = in_situ_df[target_col]

    random.seed(random_seed)
    train_idx = random.sample(list(in_situ_df[index_col].unique()), train_samples)
    test_idx = list(set(in_situ_df[index_col].unique()) - set(train_idx))

    train_idx = X[X[index_col].isin(train_idx)].index
    test_idx = X[X[index_col].isin(test_idx)].index
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_set = pd.DataFrame(X_train).join(pd.DataFrame(y_train))
    holdout_set = pd.DataFrame(X_test).join(pd.DataFrame(y_test))

    # print("Saving in situ training set to {}".format(os.path.join(root_path, path, filename[:-4]+"_{}_{}_train.csv".format(train_samples, random_seed))))
    # train_set.to_csv(os.path.join(root_path, path, filename[:-4]+"_{}_{}_train.csv".format(train_samples, random_seed)))

    # print("Saving in situ holdout set to {}".format(os.path.join(root_path, path, filename[:-4]+"_{}_{}_holdout.csv".format(train_samples, random_seed))))
    # holdout_set.to_csv(os.path.join(root_path, path, filename[:-4]+"_{}_{}_holdout.csv".format(train_samples, random_seed)))
    return train_set, holdout_set

# if __name__ == "__main__":
#     split_in_situ_data(config)