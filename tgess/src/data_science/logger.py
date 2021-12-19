import numpy as np
import pandas as pd
import time
import os
from tqdm.auto import tqdm
from .logger import *
from .helper_functions import *
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

tqdm.pandas()

# Read config file
config = read_config()

# Get root folder of project
root_path = (Path(__file__).parent / ".." / ".." ).resolve()

def standard_logging(y_train, y_train_pred, y_test, y_test_pred, model_type, model_config, ensemble, \
                     experiment_name, tune_time, train_time, test_time, config=config, experiment="standard"):
    
    """
    Function to log experiment results in a .csv file.
    
    Args:
        y_true (np array): Real target values
        y_pred (np array): Predicted target values
        in_situ_dataset (str): Filename of used in situ dataset
        simulation_dataset (str): Filename of used simulation dataset 
        simulation_samples (int): Number of simulation samples
        simulation_config (str): Name of configuration used for creating simulation dataset
        model_type (str): Name of model type used (e.g. "RandomForest")
        model_config (dict): Hyperparameters used for final model
        model_file (str): Filename of saved fitted model
        experiment_name (str): Describes which kind of experiment was done (train on in situ / train on simulation / train on both)
        tune_time (float): Time taken for training and tuning model on training set during hyperparameter tuning
        train_time (float): Time taken for final training on training set
        test_time (float): Time taken for evaluating on holdout set

    Returns:
        results (pd DataFrame): One row of a table describing logged values.
    """
    
    metrics = config["experiment"]["metrics"]
    log_path = config["experiment"]["logging"]["path"]
    log_filename = config["experiment"]["logging"]["filename"]

    in_situ_dataset = config["experiment"]["data"]["in_situ_dataset"]
    simulation_dataset = config["experiment"]["data"]["simulation_dataset"]
    simulation_samples = config["experiment"]["data"]["simulation_samples"]
    simulation_config = config["experiment"]["data"]["simulation_config"]
    
    results = {}
    results["in_situ_dataset"] = str(in_situ_dataset)
    results["simulation_dataset"] = str(simulation_dataset)
    results["simulation_samples"] = simulation_samples
    results["simulation_config"] = str(simulation_config)

    try:
        # Preferably use a human readable string as model type
        results["model_type"] = str(model_type.__name__)
    except:
        # Otherwise just use the class type as a string
        results["model_type"] = str(model_type)

    if type(model_config) == dict:
        model_config = {k:str(v) for k,v in model_config.items()}

    results["model_config"] = str(model_config)
    results["ensemble"] = str(ensemble)
    results["experiment_name"] = str(experiment_name)

    for metric in metrics:
        results["train_{}".format(metric)] = globals()[metric](y_train, y_train_pred)
        results["test_{}".format(metric)] = globals()[metric](y_test, y_test_pred)

    results["tune_time"] = tune_time
    results["train_time"] = train_time
    results["test_time"] = test_time

    ensemble_size = None

    try:
        ensemble_size = model_config["ensemble_size"]
    except:
        ensemble_size = None
    

    log_filename = str(log_filename)+"_{}.csv".format(experiment_name)


    out_file = (root_path / log_path / log_filename).resolve()
    results = {k:[v] for k, v in results.items()}
    # out_file = os.path.join(log_path, log_filename)

    saved = False
    while (saved==False):
        try:
            
            if os.path.exists(out_file):
                temp = pd.read_csv(out_file)
                run_id = np.max(temp["run_id"])+1
                results["run_id"] = run_id
                results = pd.DataFrame(results)
                results.to_csv(out_file, mode='a', header=False)
            else:
                results["run_id"] = 0
                results = pd.DataFrame(results)
                results.to_csv(out_file)

            saved = True

        except Exception as e:
            print(e)
            time.sleep(30)
    
    print("Saved log data for {} in {}".format(model_type, out_file))
    for metric in metrics:
        print("{}: {}".format(metric, results["train_{}".format(metric)].values))
        print("{}: {}".format(metric, results["test_{}".format(metric)].values))
    return results, log_filename

def create_plot_log_data(y_true, y_pred, log_filename, train):
    """
    Function to save results in order to create plots.
    
    Args:
        y_true (np array): Real target values
        y_pred (np array): Predicted target values
        run_id (int): Unique identifier for current run

    Returns:
        results (pd DataFrame): One row of a table describing logged values.
    """

    # Results per row

    # print(y_true.shape)
    # print(y_pred.shape)

    # Disabled for now
    return None


    out_df = pd.DataFrame({"target":y_true, "predicted_target":y_pred.reshape(-1,)})
    out_df["error"] = out_df["target"] - out_df["predicted_target"]
    out_df["abs_error"] = np.abs(out_df["error"])

    log_path = config["experiment"]["logging"]["path"]

    if train:
        suffix = "train"

    else:
        suffix = "test"

    out_filename = "{}_{}_predictions.csv".format(log_filename[:-4], suffix)

    out_file = (root_path / log_path / log_filename).resolve()

    out_file = (root_path / log_path / "tmp" / out_filename).resolve()
    # out_file = os.path.join(log_path, "tmp", out_filename)
    out_file = uniquify(out_file)

    out_df.to_csv(out_file)
    print("Saved plot log data in {}".format(out_file))

    return out_df

