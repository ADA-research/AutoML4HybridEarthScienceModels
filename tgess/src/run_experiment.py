import argparse
import os
import sys
import urllib.request
import zipfile

from data_science.pipeline import *
from data_science.experiment import *
from data_science.helper_functions import *
from data_science.split_data import *

tqdm.pandas()

def get_data_sim_only(config, random_seed):
    data_path = config["experiment"]["data"]["path"]
    train_samples = config["experiment"]["split"]["train_samples"]

    simulation_file = config["experiment"]["data"]["simulation_dataset"]
    simulation_file = (root_path / data_path / simulation_file).resolve()

    in_situ_file = config["experiment"]["data"]["in_situ_dataset"]
    in_situ_file = (root_path / data_path / in_situ_file).resolve()

    train_df = pd.read_csv(simulation_file)
    validation_df = None #Auto-Sklearn will automatically use a 70-30 split in this case for training/validating
    holdout_df = pd.read_csv(in_situ_file)

    return train_df, validation_df, holdout_df

def get_data_in_situ_only(config, random_seed, ts=False):
    data_path = config["experiment"]["data"]["path"]
    train_samples = config["experiment"]["split"]["train_samples"]

    in_situ_file = config["experiment"]["data"]["in_situ_dataset"]
    in_situ_file = (root_path / data_path / in_situ_file).resolve()

    in_situ_train_file = config["experiment"]["data"]["in_situ_dataset"][:-4]+"_{}_{}_train.csv".format(train_samples, random_seed)
    in_situ_train_file = (root_path / data_path / in_situ_train_file).resolve()

    in_situ_holdout_file = config["experiment"]["data"]["in_situ_dataset"][:-4]+"_{}_{}_holdout.csv".format(train_samples, random_seed)
    in_situ_holdout_file = (root_path / data_path / in_situ_holdout_file).resolve()

    if ts:
        train_df, holdout_df = split_in_situ_time_series_data(config, random_seed=random_seed)

    else:
        train_df, holdout_df = split_in_situ_data(config, random_seed=random_seed)

    validation_df = None #Auto-Sklearn will automatically use a 70-30 split in this case for training/validating
    # holdout_df = pd.read_csv(in_situ_holdout_file)

    return train_df, validation_df, holdout_df

def get_data_both(config, random_seed, ts=False):
    data_path = config["experiment"]["data"]["path"]
    train_samples = config["experiment"]["split"]["train_samples"]

    simulation_file = config["experiment"]["data"]["simulation_dataset"]
    simulation_file = (root_path / data_path / simulation_file).resolve()

    in_situ_file = config["experiment"]["data"]["in_situ_dataset"]
    in_situ_file = (root_path / data_path / in_situ_file).resolve()

    in_situ_train_file = config["experiment"]["data"]["in_situ_dataset"][:-4]+"_{}_{}_train.csv".format(train_samples, random_seed)
    in_situ_train_file = (root_path / data_path / in_situ_train_file).resolve()

    in_situ_holdout_file = config["experiment"]["data"]["in_situ_dataset"][:-4]+"_{}_{}_holdout.csv".format(train_samples, random_seed)
    in_situ_holdout_file = (root_path / data_path / in_situ_holdout_file).resolve()

    train_df = pd.read_csv(simulation_file)
    
    if ts:
        validation_df, holdout_df = split_in_situ_time_series_data(config, random_seed=random_seed)

    else:
        validation_df, holdout_df = split_in_situ_data(config, random_seed=random_seed)

    return train_df, validation_df, holdout_df

def main():
    # Create the parser
    arg_parser = argparse.ArgumentParser(description='Run GBOV experiment.')

    arg_parser.add_argument('--config',
                           metavar='config',
                           type=str,
                           nargs="?",
                           const="default",
                           default="default",
                           help="""Name of configuration file without file extension. \n
                            Default: default
                            """
                        )

    arg_parser.add_argument('--model',
                           metavar='model',
                           type=str,
                           nargs="?",
                           const="autosklearn-1",
                           default="autosklearn-1",
                           help="""Name of model configuration (see models.py). \n
                            Default: autosklearn-1
                            """
                        )

    arg_parser.add_argument('--experiment',
                           metavar='experiment',
                           type=str,
                           nargs="?",
                           const="standard",
                           default="standard",
                           help="""Type of experiment to perform, standard or custom (proposed method). \n
                            Default: standard
                            """
                        )

    arg_parser.add_argument('--data',
                           metavar='data',
                           type=str,
                           nargs="?",
                           const="in_situ",
                           default="in_situ",
                           help="""Type of data to use for experiment. \n
                            Options: \n
                            - simulation \n
                            - in_situ (default) \n
                            - both \n
                            """
                        )

    arg_parser.add_argument('--random_seed',
                           metavar='random_seed',
                           type=int,
                           nargs="?",
                           const=1,
                           default=1,
                           help="""Random seed to use for experiment. \n
                                Default: 1
                                """)

    # Execute the parse_args() method
    args = arg_parser.parse_args()

    config = args.config
    model_setup = args.model
    experiment_type = args.experiment
    data = args.data
    random_seed = args.random_seed

    root_path = (Path(__file__).parent).resolve()
    config_path = os.path.join(root_path, "data_science")

    # Read config file
    config = read_config(config_path, config)
    config["experiment"]["model_config"] = model_setup

    # Download files if necessary
    data_path = config["experiment"]["data"]["path"]
    in_situ_file = config["experiment"]["data"]["in_situ_dataset"]
    in_situ_file = (root_path / ".." / data_path / in_situ_file).resolve()
    in_situ_train_file = str(in_situ_file)[:-4]+"_{}_train.csv"


    simulation_file = config["experiment"]["data"]["simulation_dataset"]
    simulation_file = (root_path / ".." / data_path / in_situ_file).resolve()
    
    if not os.path.isfile(in_situ_file):
        url = config["experiment"]["data"]["url"]
        extract_dir = os.path.join(root_path, "..", data_path)

        zip_path, _ = urllib.request.urlretrieve(url)

        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(extract_dir)

    models = get_models(model_setup=model_setup, config=config)

    

    feature_cols = config["experiment"]["data"]["feature_cols"]
    experiment_name = config["experiment"]["name"]

    # Get root folder of project
    root_path = (Path(__file__).parent / "..").resolve()

    if data=="simulation":
        target_col = config["experiment"]["data"]["simulation_target_col"]
        train_df, validation_df, holdout_df = get_data_sim_only(config, random_seed)

    elif data=="in_situ":
        target_col = config["experiment"]["data"]["in_situ_target_col"]
        train_df, validation_df, holdout_df = get_data_in_situ_only(config, random_seed)

    elif data=="in_situ_ts":
        target_col = [config["experiment"]["data"]["in_situ_target_col"], config["experiment"]["data"]["in_situ_target_col"]]
        train_df, validation_df, holdout_df = get_data_in_situ_only(config, random_seed, ts=True)

    elif data=="both_ts":
        target_col = [config["experiment"]["data"]["simulation_target_col"], config["experiment"]["data"]["in_situ_target_col"]]
        train_df, validation_df, holdout_df = get_data_both(config, random_seed, ts=True)

    else:
        target_col = [config["experiment"]["data"]["simulation_target_col"], config["experiment"]["data"]["in_situ_target_col"]]
        train_df, validation_df, holdout_df = get_data_both(config, random_seed)

    if "extra_feature_cols" in list(config["experiment"]["data"].keys()):
        extra_cols = config["experiment"]["data"]["extra_feature_cols"]

    else:
        extra_cols = []

    if model_setup == "proposed_ensemble":
        config["models"]["ensemble_size"]=5

    if experiment_type == "custom":
        results, plot_data = custom_experiment(models, target_col, feature_cols, train_df, validation_df, holdout_df, experiment_name="custom", extra_cols=extra_cols, config=config)

    elif experiment_type == "ts":
        if data=="in_situ_ts":
            results, plot_data = standard_time_series_experiment(models, target_col, feature_cols, train_df, validation_df, holdout_df, experiment_name="standard", extra_cols=extra_cols, config=config)

        else:
            results, plot_data = custom_time_series_experiment(models, target_col, feature_cols, train_df, validation_df, holdout_df, experiment_name="custom", extra_cols=extra_cols, config=config)

    else:
        results, plot_data = standard_experiment(models, target_col, feature_cols, train_df, validation_df, holdout_df, experiment_name="standard", extra_cols=extra_cols, config=config)

if __name__ == "__main__":
    main()