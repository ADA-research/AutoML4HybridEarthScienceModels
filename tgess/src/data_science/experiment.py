import numpy as np
import pandas as pd
import os
import time

from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.model_selection import cross_validate, cross_val_score, PredefinedSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            r2_score, mean_absolute_percentage_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, RationalQuadratic
from modAL.models import ActiveLearner

from .helper_functions import *
from .logger import *
from .models import *

tqdm.pandas()

# Read config file
config = read_config()

tmp_folder = "/local/s1281437/tmp_"

if not os.path.exists("/local/s1281437"):
    tmp_folder = "/scratch/s1281437/tmp_"

# Get root folder of project
root_path = (Path(__file__).parent / ".." / ".." ).resolve()

def perform_simulation_cv(data, model_type, hyperparams, n_folds=5, target_col="lai", \
                          feature_cols=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', \
                          'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'solar_zenith', 'observer_zenith', 'relative_azimuth'], config=config):
    """
    Train and test model on simulation data only, using a standard K-fold cross-validation approach.
    
    Args:
        data (pd DataFrame): Simulation dataset
        model_type (sklearn model): Regression model type to use.
        hyperparams (dict): Hyperparameters to tune
        n_folds (int): Number of cross-validation folds
        target_col (str): Column name of target variable
        feature_cols (list of str): Column names of feature variables

    Returns:
        cv_scores (dict): Dictionary describing cross-validated scores of several different metrics.
    """
    
    X = data[feature_cols]
    y = data[target_col]
    
    model = initialize_model(model_type, hyperparams)
    
    metrics = config["experiment"]["metrics"]
    cv_scores = cross_validate(model, X, y, cv=n_folds, scoring=metrics)
    
    cv_scores = {key:np.mean(value) for key, value in cv_scores.items()}

    cv_scores["model"] = str(model_type.__name__)
    cv_scores["hyperparams"] = hyperparams
                             
    
    return cv_scores

def get_holdout_set_predictions(train_data, test_data, model_type, hyperparams, target_col="LAI_Warren", \
                          feature_cols=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', \
                          'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'solar_zenith', 'observer_zenith', 'relative_azimuth'], extra_cols=[], standardize=True, config=config):
    """
    Simple function to train a model on a training set and test a model on a separate holdout set.
    
    Args:
        train_data (pd DataFrame): Training dataset
        test_data (pd DataFrame): Testing dataset
        model_type (sklearn model): Regression model type to use.
        hyperparams (dict): Model hyperparameters
        target_cols (tuple of str): Column name of target variable in train_data and test_date
        feature_cols (list of str): Column names of feature variables
        standardize (bool): Whether to standardize the feature values using sklearn's StandardScaler().

    Returns:
        cv_scores (dict): Dictionary describing cross-validated scores of several different metrics.
    """
    
    # Train on entire training set

    # if type(target_col) == list or type(target_col) == tuple:
    #     train_target_col = target_col[0]
    #     test_target_col = target_col[1]

    # else:
    train_target_col = target_col
    test_target_col = target_col

    X_train = np.array(train_data[feature_cols])
    y_train = np.array(train_data[train_target_col]) 

    # Shuffle training set
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)

    X_train = X_train[idx]
    y_train = y_train[idx]

    X_test = np.array(test_data[feature_cols])
    y_test = np.array(test_data[test_target_col]) 

    # Standardize data
    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Fit model on entire training set
    t_start = time.process_time()

    # Active learning models are trained slightly differently
    if model_type == ActiveLearner: 
        model = train_active_learner(X_train, y_train, **hyperparams)

    else:
        if model_type == AutoMLStackingRegressor:
            # Columns which were not present in simulation data but are used as features in in_situ data
            hyperparams["extra_cols"] = [train_data[feature_cols].columns.get_loc(col) for col in extra_cols]
            # print(hyperparams["extra_cols"])
            # print(train_data[feature_cols].columns)
            # print(X_train.shape)

        model = initialize_model(model_type, hyperparams)
        model.fit(X_train, y_train)
        # model.refit(X_train, y_train)
        try:
            print(model.sprint_statistics())
        except:
            print("Finished model fitting")

    # Get training set predictions  
    y_train_pred = model.predict(X_train)

    train_time = time.process_time()-t_start

    # Get test set predictions
    t_start = time.process_time()
    y_test_pred = model.predict(X_test)
    test_time = time.process_time()-t_start
    
    return model, y_train, y_train_pred, y_test, y_test_pred, train_time, test_time

def custom_time_series_experiment(models, target_col, feature_cols, train_df, validation_df, holdout_df, experiment_name, extra_cols=[], config=config):
    """
    Performs an experiment for each model included in models argument.
    Creates .csv files with results.
    
    Args:
        models (list of dict): List of dictionaries, where each dictionary describes a model and its hyperparameters. 
                               E.g.: 
                               [
                                    {
                                        "model_type":RandomForestRegressor
                                        "hyperparams":
                                        {
                                            "n_estimators": 100
                                            "max_depth": "auto"
                                        }
                                    },

                                    {
                                        ...
                                    }

                               ]
        target_cols (tuple of str): Column name of target variable in train_data and test_date
        feature_cols (list of str): Column names of feature variables
        train_df (pd DataFrame): Data used for training
        validation_df (pd DataFrame):
        holdout_df (pd DataFrame): 

    Returns:
        results (pd DataFrame): Table describing logged values such as model type, hyperparameters and performance.
        plot_data (pd DataFrame): Table describing errors per predicted sample, used for several plotting functions.

    """    
    
    timestamp = datetime.now()
    model_file = None

    for model in tqdm(models, desc="Training models"):
        model_type = model["model_type"]
        hyperparams = model["hyperparams"]
        if "model_name" in model.keys():
            model_name = model["model_name"]
        else:
            model_name = None
        t_start = time.process_time()


        if validation_df is not None:
            # model["hyperparams"]["resampling_strategy"] = PredefinedSplit(test_fold=np.array(range(len(train_df), len(train_df)+len(validation_df))))
            if type(target_col) == list:
                all_cols = feature_cols+flatten_list(target_col)
            elif type(target_col) == tuple:
                all_cols = feature_cols+flatten_list(list(target_col))
            else:
                all_cols = feature_cols+[target_col]
            # train_df = (train_df[all_cols].append(validation_df[all_cols])).reindex()
            train_df = train_df[all_cols]
            # train_df.to_csv("train_bug.csv")
            # holdout_df[all_cols].to_csv("holdout_bug.csv")


        tune_time = time.process_time()-t_start

        print("Training auto-sklearn ensemble")


        # Train on training set (simulation), test on holdout set
        model, y_train, y_train_pred, y_test, y_test_pred, train_time, test_time = get_holdout_set_predictions(train_df, train_df, model_type, hyperparams, \
                                         target_col[0], feature_cols, extra_cols=[])

        if model_type == AutoSklearnRegressor:
            ensemble = model.show_models()
        else:
            ensemble = None

        # results = standard_logging(y_train, y_train_pred, y_test, y_test_pred, model_type, hyperparams, ensemble, \
        #                  experiment_name, tune_time, train_time, test_time)

        # train_plot_data = create_plot_log_data(y_train, y_train_pred, log_file)
        # plot_data = create_plot_log_data(y_test, y_test_pred, log_file)

        # Train the stacked estimator by reusing the estimators from the autosklearn ensemble as input features
        if "ensemble_size" in config["models"].keys():
            ensemble_size = config["models"]["ensemble_size"]
        else:
            ensemble_size = 1

        final_estimator_params = {
            "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"], 
            "per_run_time_limit": config["models"]["auto_sklearn"]["time_per_task"],
            "initial_configurations_via_metalearning":1,
            "ensemble_size":ensemble_size,
            "ensemble_nbest":ensemble_size,
            "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
            "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
            "tmp_folder":uniquify_folder(tmp_folder),
            "delete_tmp_folder_after_terminate":True,
            "include_estimators":["adaboost", "ard_regression", "decision_tree", \
                                  "extra_trees", "gaussian_process", "gradient_boosting", "k_nearest_neighbors", \
                                  "liblinear_svr", "libsvm_svr", "mlp", "random_forest", "sgd"]
        }

        final_estimator = AutoSklearnRegressor

        hyperparams = {"automl":model, "final_estimator":final_estimator}

        print("Training AutoMLStackingRegressor - time series")
        estimated_target_cols = []

        # Create columns for inversion estimates
        if target_col[0] != target_col[1]:
            if type(target_col[0])==list:
                estimated_target_cols = [str(col)+"_estim" for col in target_col[0]]
                # print(estimated_target_cols)
                validation_df[estimated_target_cols] = pd.DataFrame(model.predict(validation_df[feature_cols]), columns=estimated_target_cols)
                holdout_df[estimated_target_cols] = pd.DataFrame(model.predict(holdout_df[feature_cols]), columns=estimated_target_cols)
            else:
                estimated_target_cols = [target_col[0]+"_estim"]
                validation_df[target_col[0]+"_estim"] = pd.DataFrame(model.predict(validation_df[feature_cols]), columns=estimated_target_cols)
                holdout_df[target_col[0]+"_estim"] = pd.DataFrame(model.predict(holdout_df[feature_cols]), columns=estimated_target_cols)

        # Convert to time series
        validation_df = flatten_time_series(validation_df, feature_cols+extra_cols+estimated_target_cols, target_col[1])
        holdout_df = flatten_time_series(holdout_df, feature_cols+extra_cols+estimated_target_cols, target_col[1])
        time_series_feature_cols = get_time_series_cols(validation_df, feature_cols+extra_cols+estimated_target_cols, config=config)

        # Train on validation set (in situ), test on holdout set
        model, y_train, y_train_pred, y_test, y_test_pred, train_time, test_time = get_holdout_set_predictions(validation_df, holdout_df, final_estimator, final_estimator_params, \
                                         target_col[1], time_series_feature_cols, extra_cols=extra_cols)

        try:
            ensemble = [ensemble, model.final_estimator.show_models()]
        except:
            pass

        results, log_file = standard_logging(y_train, y_train_pred, y_test, y_test_pred, model_name, hyperparams, ensemble, \
                         experiment_name, tune_time, train_time, test_time, config=config)

        train_plot_data = create_plot_log_data(y_train, y_train_pred, log_file, train=True)
        plot_data = create_plot_log_data(y_test, y_test_pred, log_file, train=False)

    return results, plot_data

def custom_experiment(models, target_col, feature_cols, train_df, validation_df, holdout_df, experiment_name, extra_cols=[], config=config):
    """
    Performs an experiment for each model included in models argument.
    Creates .csv files with results.
    
    Args:
        models (list of dict): List of dictionaries, where each dictionary describes a model and its hyperparameters. 
                               E.g.: 
                               [
                                    {
                                        "model_type":RandomForestRegressor
                                        "hyperparams":
                                        {
                                            "n_estimators": 100
                                            "max_depth": "auto"
                                        }
                                    },

                                    {
                                        ...
                                    }

                               ]
        target_cols (tuple of str): Column name of target variable in train_data and test_date
        feature_cols (list of str): Column names of feature variables
        train_df (pd DataFrame): Data used for training
        validation_df (pd DataFrame):
        holdout_df (pd DataFrame): 

    Returns:
        results (pd DataFrame): Table describing logged values such as model type, hyperparameters and performance.
        plot_data (pd DataFrame): Table describing errors per predicted sample, used for several plotting functions.

    """    
    
    timestamp = datetime.now()
    model_file = None

    for model in tqdm(models, desc="Training models"):
        model_type = model["model_type"]
        if "model_name" in model.keys():
            model_name = model["model_name"]
        else:
            model_name = None
        hyperparams = model["hyperparams"]

        t_start = time.process_time()

        if validation_df is not None:
            # model["hyperparams"]["resampling_strategy"] = PredefinedSplit(test_fold=np.array(range(len(train_df), len(train_df)+len(validation_df))))
            if type(target_col) == list:
                all_cols = feature_cols+target_col
            elif type(target_col) == tuple:
                all_cols = feature_cols+list(target_col)
            else:
                all_cols = feature_cols+[target_col]
            # train_df = (train_df[all_cols].append(validation_df[all_cols])).reindex()
            # train_df = train_df[all_cols]
            # train_df.to_csv("train_bug.csv")
            # holdout_df[all_cols].to_csv("holdout_bug.csv")

        tune_time = time.process_time()-t_start

        print("Training auto-sklearn ensemble")

        # Train on training set (simulation), test on holdout set
        model, y_train, y_train_pred, y_test, y_test_pred, train_time, test_time = get_holdout_set_predictions(train_df, train_df, model_type, hyperparams, \
                                         target_col[0], feature_cols, extra_cols=[])

        if model_type == AutoSklearnRegressor:
            ensemble = model.show_models()
        else:
            ensemble = None

        # results = standard_logging(y_train, y_train_pred, y_test, y_test_pred, model_type, hyperparams, ensemble, \
        #                  experiment_name, tune_time, train_time, test_time)

        # train_plot_data = create_plot_log_data(y_train, y_train_pred, log_file)
        # plot_data = create_plot_log_data(y_test, y_test_pred, log_file)

        # Train the stacked estimator by reusing the estimators from the autosklearn ensemble as input features
        if model_name == "Proposed ensemble":
            ensemble_size = 5
        else:
            ensemble_size = 1

        final_estimator_params = {
            "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"], 
            "per_run_time_limit": config["models"]["auto_sklearn"]["time_per_task"],
            "initial_configurations_via_metalearning":ensemble_size,
            "ensemble_size":ensemble_size,
            "ensemble_nbest":ensemble_size,
            "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
            "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
            "tmp_folder":uniquify_folder(tmp_folder),
            "max_models_on_disc":ensemble_size,
            "delete_tmp_folder_after_terminate":False,
            "include_estimators":["adaboost", "ard_regression", "decision_tree", \
                                  "extra_trees", "gaussian_process", "gradient_boosting", "k_nearest_neighbors", \
                                  "liblinear_svr", "libsvm_svr", "mlp", "random_forest", "sgd"]
        }

        final_estimator = AutoSklearnRegressor(**final_estimator_params)

        hyperparams = {"automl":model, "final_estimator":final_estimator}

        print("Training AutoMLStackingRegressor")

        # Train on validation set (in situ), test on holdout set
        model, y_train, y_train_pred, y_test, y_test_pred, train_time, test_time = get_holdout_set_predictions(validation_df, holdout_df, AutoMLStackingRegressor, hyperparams, \
                                         target_col[1], feature_cols+extra_cols, extra_cols=extra_cols)

        try:
            ensemble = [ensemble, model.final_estimator.show_models()]
        except:
            pass

        results, log_file = standard_logging(y_train, y_train_pred, y_test, y_test_pred, model_name, hyperparams, ensemble, \
                         experiment_name, tune_time, train_time, test_time, config=config)

        train_plot_data = create_plot_log_data(y_train, y_train_pred, log_file, train=True)
        plot_data = create_plot_log_data(y_test, y_test_pred, log_file, train=False)

    return results, plot_data

def standard_time_series_experiment(models, target_col, feature_cols, train_df, validation_df, holdout_df, experiment_name, extra_cols=[], config=config):
    """
    Performs an experiment for each model included in models argument.
    Creates .csv files with results.
    
    Args:
        models (list of dict): List of dictionaries, where each dictionary describes a model and its hyperparameters. 
                               E.g.: 
                               [
                                    {
                                        "model_type":RandomForestRegressor
                                        "hyperparams":
                                        {
                                            "n_estimators": 100
                                            "max_depth": "auto"
                                        }
                                    },

                                    {
                                        ...
                                    }

                               ]
        target_cols (tuple of str): Column name of target variable in train_data and test_date
        feature_cols (list of str): Column names of feature variables
        train_df (pd DataFrame): Data used for training
        validation_df (pd DataFrame):
        holdout_df (pd DataFrame): 

    Returns:
        results (pd DataFrame): Table describing logged values such as model type, hyperparameters and performance.
        plot_data (pd DataFrame): Table describing errors per predicted sample, used for several plotting functions.

    """    
    
    timestamp = datetime.now()
    model_file = None

    for model in tqdm(models, desc="Training models"):
        model_type = model["model_type"]
        if "model_name" in model.keys():
            model_name = model["model_name"]
        else:
            model_name = None
        hyperparams = model["hyperparams"]

        # if model_type == ShapedNeuralNetwork:
        #     hyperparams["n_inputs"] = len(feature_cols)

        
        t_start = time.process_time()

        # if model_type == AutoSklearnRegressor:
        #     if validation_df is not None:
        #         model["hyperparams"]["resampling_strategy"] = PredefinedSplit(test_fold=np.array(range(len(train_df), len(train_df)+len(validation_df))))
        #         train_df = (train_df.append(validation_df)).reindex()

        tune_time = time.process_time()-t_start

        # Convert to time series
        train_df = flatten_time_series(train_df, feature_cols+extra_cols, target_col[0])
        # validation_df = flatten_time_series(validation_df, feature_cols+extra_cols, target_col)
        holdout_df = flatten_time_series(holdout_df, feature_cols+extra_cols, target_col[0])
        time_series_feature_cols = get_time_series_cols(train_df, feature_cols+extra_cols, config=config)

        # Train on validation set (in situ), test on holdout set

        print("Target col = ", target_col[0])
        model, y_train, y_train_pred, y_test, y_test_pred, train_time, test_time = get_holdout_set_predictions(train_df, holdout_df, model_type, hyperparams, \
                                         target_col[0], time_series_feature_cols, extra_cols=extra_cols)

        if model_type == AutoSklearnRegressor:
            ensemble = model.show_models()
        else:
            ensemble = None

        results, log_file = standard_logging(y_train, y_train_pred, y_test, y_test_pred, model_name, hyperparams, ensemble, \
                         experiment_name, tune_time, train_time, test_time, config=config)

        # train_plot_data = create_plot_log_data(y_train, y_train_pred, log_file, train=True)
        # plot_data = create_plot_log_data(y_test, y_test_pred, log_file, train=False)

    return results, None


def standard_experiment(models, target_col, feature_cols, train_df, validation_df, holdout_df, experiment_name, extra_cols=[], config=config):
    """
    Performs an experiment for each model included in models argument.
    Creates .csv files with results.
    
    Args:
        models (list of dict): List of dictionaries, where each dictionary describes a model and its hyperparameters. 
                               E.g.: 
                               [
                                    {
                                        "model_type":RandomForestRegressor
                                        "hyperparams":
                                        {
                                            "n_estimators": 100
                                            "max_depth": "auto"
                                        }
                                    },

                                    {
                                        ...
                                    }

                               ]
        target_cols (tuple of str): Column name of target variable in train_data and test_date
        feature_cols (list of str): Column names of feature variables
        train_df (pd DataFrame): Data used for training
        validation_df (pd DataFrame):
        holdout_df (pd DataFrame): 

    Returns:
        results (pd DataFrame): Table describing logged values such as model type, hyperparameters and performance.
        plot_data (pd DataFrame): Table describing errors per predicted sample, used for several plotting functions.

    """    
    
    timestamp = datetime.now()
    model_file = None

    for model in tqdm(models, desc="Training models"):
        model_type = model["model_type"]
        if "model_name" in model.keys():
            model_name = model["model_name"]
        else:
            model_name = None
        hyperparams = model["hyperparams"]

        # if model_type == ShapedNeuralNetwork:
        #     hyperparams["n_inputs"] = len(feature_cols)

        
        t_start = time.process_time()

        if model_type == AutoSklearnRegressor:
            if validation_df is not None:
                model["hyperparams"]["resampling_strategy"] = PredefinedSplit(test_fold=np.array(range(len(train_df), len(train_df)+len(validation_df))))
                train_df = (train_df.append(validation_df)).reindex()

        tune_time = time.process_time()-t_start

        model, y_train, y_train_pred, y_test, y_test_pred, train_time, test_time = get_holdout_set_predictions(train_df, holdout_df, model_type, hyperparams, \
                                         target_col, feature_cols+extra_cols, extra_cols=extra_cols)

        if model_type == AutoSklearnRegressor:
            ensemble = model.show_models()
        else:
            ensemble = None

        results, log_file = standard_logging(y_train, y_train_pred, y_test, y_test_pred, model_name, hyperparams, ensemble, \
                         experiment_name, tune_time, train_time, test_time, config=config)

        train_plot_data = create_plot_log_data(y_train, y_train_pred, log_file, train=True)
        plot_data = create_plot_log_data(y_test, y_test_pred, log_file, train=False)

    return results, plot_data

def simulation_only_experiment(models, target_col, feature_cols, config=config):
    """
    Performs an experiment for each model included in models argument. For these experiments, each model is trained on simulation data only. 
    Each model's performance is tested on a holdout set of in situ data (e.g. 30% stratified random sample).
    Creates .csv files with results.
    
    Args:
        models (list of dict): List of dictionaries, where each dictionary describes a model and its hyperparameters. 
                               E.g.: 
                               [
                                    {
                                        "model_type":RandomForestRegressor
                                        "hyperparams":
                                        {
                                            "n_estimators": 100
                                            "max_depth": "auto"
                                        }
                                    },

                                    {
                                        ...
                                    }

                               ]
        target_cols (tuple of str): Column name of target variable in train_data and test_date
        feature_cols (list of str): Column names of feature variables

    Returns:
        results (pd DataFrame): Table describing logged values such as model type, hyperparameters and performance.
        plot_data (pd DataFrame): Table describing errors per predicted sample, used for several plotting functions.

    """     
        
    data_path = config["experiment"]["data"]["path"]

    in_situ_file = config["experiment"]["data"]["in_situ_dataset"]
    in_situ_file = (root_path / data_path / in_situ_file).resolve()

    in_situ_holdout_file = config["experiment"]["data"]["in_situ_dataset"][:-4]+"_holdout.csv"
    in_situ_holdout_file = (root_path / data_path / in_situ_holdout_file).resolve()

    simulation_file = config["experiment"]["data"]["simulation_dataset"]
    simulation_file = (root_path / data_path / simulation_file).resolve()

    simulation_df = pd.read_csv(simulation_file)
    in_situ_holdout_df = pd.read_csv(in_situ_holdout_file)
    

    results, plot_data = standard_experiment(models, target_col, feature_cols, train_df=simulation_df, validation_df=None, holdout_df=in_situ_holdout_df, experiment_name="simulation_only")

    return results, plot_data

def in_situ_only_experiment(models, target_col, feature_cols, config=config):
    """
    Performs an experiment for each model included in models argument. For these experiments, each model is trained on in situ data only (70% stratified random sample). 
    Each model's performance is tested on a holdout set of in situ data (30% stratified random sample).
    Creates .csv files with results.
    
    Args:
        models (list of dict): List of dictionaries, where each dictionary describes a model and its hyperparameters. 
                               E.g.: 
                               [
                                    {
                                        "model_type":RandomForestRegressor
                                        "hyperparams":
                                        {
                                            "n_estimators": 100
                                            "max_depth": "auto"
                                        }
                                    },

                                    {
                                        ...
                                    }

                               ]
        target_cols (tuple of str): Column name of target variable in train_data and test_date
        feature_cols (list of str): Column names of feature variables

    Returns:
        results (pd DataFrame): Table describing logged values such as model type, hyperparameters and performance.
        plot_data (pd DataFrame): Table describing errors per predicted sample, used for several plotting functions.

    """    
    
    data_path = config["experiment"]["data"]["path"]

    in_situ_file = config["experiment"]["data"]["in_situ_dataset"]
    in_situ_file = (root_path / data_path / in_situ_file).resolve()

    in_situ_train_file = config["experiment"]["data"]["in_situ_dataset"][:-4]+"_train.csv"
    in_situ_train_file = (root_path / data_path / in_situ_train_file).resolve()

    in_situ_holdout_file = config["experiment"]["data"]["in_situ_dataset"][:-4]+"_holdout.csv"
    in_situ_holdout_file = (root_path / data_path / in_situ_holdout_file).resolve()

    in_situ_train_df = pd.read_csv(in_situ_train_file)
    in_situ_holdout_df = pd.read_csv(in_situ_holdout_file)
   
    results, plot_data = standard_experiment(models, target_col, feature_cols, train_df=in_situ_train_df, validation_df=None, holdout_df=in_situ_holdout_df, experiment_name="in_situ_only")

    return results, plot_data