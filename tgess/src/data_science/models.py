import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tqdm.auto import tqdm
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            r2_score, mean_absolute_percentage_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, RationalQuadratic
from sklearn.base import BaseEstimator, RegressorMixin
from modAL.models import ActiveLearner
# from tensorflow import keras
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import EqualsCondition

import sklearn.metrics
import autosklearn.regression
import autosklearn.pipeline.components.regression
from autosklearn.pipeline.components.base import AutoSklearnRegressionAlgorithm
from autosklearn.pipeline.constants import SPARSE, DENSE, \
    SIGNED_DATA, UNSIGNED_DATA, PREDICTIONS

from .helper_functions import *

# autosklearn only works on linux
if os.name != 'nt':
    from autosklearn.regression import AutoSklearnRegressor

# Read config file
config = read_config()
tmp_folder = "/local/s1281437/tmp_"

if not os.path.exists("/local/s1281437"):
    tmp_folder = "/scratch/s1281437/tmp_"

def GP_regression_std(regressor, X, batch_size):
    """
    Function query based on std predicted by regressor
    
    Args:
        regressor (sklearn model): Regression model type to use.
        X (np array): Training features
        batch_size (int): Number of samples to query.

    Returns:
        query_idx: indices of chosen samples
        X[query_idx]: values of chosen samples

    """

    _, std = regressor.predict(X, return_std=True)
    # Perform argmax for batch_size samples
    query_idx = np.argpartition(std, -batch_size)[-batch_size:]

    return query_idx, X[query_idx]


def train_active_learner(X, y, estimator, query_strategy, estimator_parameters, active_learner_parameters, kernel_parameters=None):
    """
    Function to train a generic ActiveLearner model
    
    Args:
        X (np array): Training features.
        y (np array): Training target.
        estimator (sklearn model): Regression model type to use.
        estimator_params (dict): Dictionary of estimator hyperparameters.
        query_strategy (function): Strategy to use for active learning querying.
        n_initial (int): Number of samples in initial training batch.
        n_total (int): Total number of samples to train on.
        batch_size (int): Number of samples per training batch (after first one).

    Returns:
        regressor: Trained model of type estimator.

    """

    n_initial = active_learner_parameters["n_initial"]
    n_total =  active_learner_parameters["n_total"]
    batch_size = active_learner_parameters["batch_size"]


    n_batches = int((n_total - n_initial) / batch_size)
    n_last_batch = n_total - (n_initial + n_batches * batch_size)
    
    initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
    X_training, y_training = X[initial_idx], y[initial_idx]
    

    if estimator == GaussianProcessRegressor:
        kernel = estimator_parameters.pop("kernel")
        kernel = kernel(kernel_parameters)

    # kernel = RBF(length_scale=1.9) \
    #                + WhiteKernel()

    #              + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    # kernel = RationalQuadratic(length_scale_bounds="fixed", alpha_bounds="fixed")

    regressor = ActiveLearner(
        estimator=estimator(**estimator_parameters),
        query_strategy=query_strategy, 
        X_training=X_training, y_training=y_training
    )
    
    batches = [batch_size for i in range(n_batches)]
    
    if n_last_batch>0:
        batches.append(n_last_batch)
    
    for n_queries in tqdm(batches, desc="Fitting {}".format(str(estimator))):
        query_idx, query_instance = regressor.query(X, batch_size)
        regressor.teach(X[query_idx], y[query_idx])
     
    # Randomly sample from samples that are not included in current training batch   
    # val_idx = [np.random.choice([i for i in range(0, len(y)) if i not in query_idx]) for j in range(0,batch_size)]
        
    # Debug purposes only:
    val_score = np.mean(np.square(y - regressor.predict(X)))
    print("Validation MSE: {}".format(val_score))
    
    return regressor

# class ShapedNeuralNetwork(keras.Model):
#     """
#     Class to create a shaped neural network (for shapes, see: https://mikkokotila.github.io/slate/#shapes).
#     Inherits from keras.layer.layers base class
#     """

#     def __init__(self, n_inputs, n_layers=3, neuron_max=8, activation="relu", batch_norm=True, shape="funnel", batch_size=32, epochs=10, validation_split=20.0):
#         super(ShapedNeuralNetwork, self).__init__()
#         self.n_layers = n_layers
#         self.neuron_max = neuron_max
#         self.activation = activation
#         self.batch_norm = batch_norm
#         self.shape = shape
#         self.n_inputs = n_inputs

#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.validation_split = validation_split
#         self._create_architecture()
#         super().compile(optimizer="adam", loss="mean_squared_error")
#         # super().build(input_shape=(1, 15))
#         # super().summary()

#     def _get_n_parameters(self):
#         pass

#     def _funnel(self, ):
#         neurons_per_layer = []
#         neurons = self.neuron_max

#         for i in range(self.n_layers):
#             neurons_per_layer.append(neurons)
#             neurons = math.ceil(neurons/2)

#         # self.n_parameters = 
#         return neurons_per_layer

#     def _long_funnel(self, ):
#         neurons_per_layer = []
#         neurons = self.neuron_max

#         for i in range(ceil(self.n_layers/2)):
#             neurons_per_layer.append(neurons)

#         for i in range(ceil(self.n_layers/2), self.n_layers):
#             neurons_per_layer.append(neurons)
#             neurons = math.ceil(neurons/2)
            
#         return neurons_per_layer

#     def _brick(self, ):
#         neurons_per_layer = []
#         neurons = self.neuron_max

#         for i in range(self.n_layers):
#             neurons_per_layer.append(neurons)
            
#         return neurons_per_layer

#     def _get_shape(self, shape_name):
#         shape_functions = \
#         {
#             "default": self._brick,
#             "funnel":self._funnel,
#             "long_funnel": self._long_funnel,
#             "brick": self._brick
#         }

#         if shape_name in shape_functions.keys():
#             return shape_functions[shape_name]

#         else:
#             return shape_functions["default"]

#     def _create_architecture(self):
#         self.hidden_layers = []

#         # self.hidden_layers = [keras.layers.Input(shape = (self.n_inputs,))]
#         neurons_per_layer = self._get_shape(self.shape)()

#         for neurons in neurons_per_layer:
#             self.hidden_layers.append(keras.layers.Dense(neurons, activation=self.activation))

#             if self.batch_norm:
#                 self.hidden_layers.append(keras.layers.BatchNormalization())

#         self.output_layer = keras.layers.Dense(1, activation="linear")

#     def call(self, inputs):
#         # print(self.hidden_layers)
#         x = self.hidden_layers[0](inputs)

#         for layer in self.hidden_layers[1:]:
#             x = layer(x)
        
#         x = self.output_layer(x)    
#         return x

#     def fit(self, x, y):
#         super().fit(x, y, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split, verbose=0)


class AutoMLStackingRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, automl, final_estimator, extra_cols=[]):
        """
        Works similar to sklearn's StackingRegressor, 
        except that the input estimators are not refitted during a call to fit().

        param automl: Fitted auto-sklearn object, from which estimator pipelines can be extracted
        param final_estimator: Untrained sklearn estimator
        """
        super().__init__()
        self.automl = automl
        self.final_estimator = final_estimator
        self.is_fit=False
        self.extra_cols=extra_cols
    
    def _get_estimators(self):
        """
        Extracts estimators from autosklearn object
        """
        
        self.estimators = [m[1] for m in self.automl.get_models_with_weights()]
        return
    
    def _get_X_final_model(self, X):
        """
        Adds predictions from autosklearn estimators as features in X
        """
        X_ensemble = np.copy(X)

        if len(self.extra_cols)>1:
            X_ensemble = np.delete(X_ensemble, self.extra_cols, 1)

        y_pred = np.array([estimator.predict(X_ensemble) for estimator in self.estimators])

        y_pred = y_pred.reshape(len(X), -1)

        X_final_model = np.c_[X, y_pred]  
        return X_final_model

    def fit(self, X, y):
        self._get_estimators()
        
        X = self._get_X_final_model(X)

        self.final_estimator.fit(X, y)
        self._fit=True

        return self

    def predict(self, X):
        assert(self._fit), "Unable to predict. Call the fit() function first!"

        X = self._get_X_final_model(X)

        y_pred = self.final_estimator.predict(X)

        return y_pred


class ActiveLearningGaussianProcessRegression(AutoSklearnRegressionAlgorithm):
    """
    Custom auto-sklearn regression estimator that implements Gaussian Process Regression with Active Learning and exposes hyperparameters to auto-sklearn.
    Based on: https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/regression/gaussian_process.py
    """
    
    def __init__(self, batch_size, n_total, alpha, thetaL, thetaU, random_state=None):
       
        # Active learner hyperparameters
        self.batch_size = batch_size
        self.n_total = n_total
        
        # GPR hyperparameters
        self.alpha = alpha
        self.thetaL = thetaL
        self.thetaU = thetaU
        
        # Other
        self.random_state = random_state
        self.estimator = None
        self.regressor = None
        self.scaler = None

    def fit(self, X, y):
        import sklearn.gaussian_process
        import numpy.random
        from modAL.models import ActiveLearner
        
        def GP_regression_std(regressor, X, batch_size):
            """
            Function query based on std predicted by regressor

            Args:
                regressor (sklearn model): Regression model type to use.
                X (np array): Training features
                batch_size (int): Number of samples to query.

            Returns:
                query_idx: indices of chosen samples
                X[query_idx]: values of chosen samples

            """

            _, std = regressor.predict(X, return_std=True)
            # Perform argmax for batch_size samples
            query_idx = np.argpartition(std, -batch_size)[-batch_size:]

            return query_idx, X[query_idx]

        self.alpha = float(self.alpha)
        self.thetaL = float(self.thetaL)
        self.thetaU = float(self.thetaU)

        n_features = X.shape[1]
        kernel = sklearn.gaussian_process.kernels.RBF(
            length_scale=[1.0]*n_features,
            length_scale_bounds=[(self.thetaL, self.thetaU)]*n_features)

        # Instantiate a Gaussian Process model
        self.regressor = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            optimizer='fmin_l_bfgs_b',
            alpha=self.alpha,
            copy_X_train=True,
            random_state=self.random_state,
            normalize_y=True,
            )

        if self.batch_size <= len(X):
            self.estimator = ActiveLearner(
                estimator=self.regressor,
                query_strategy=GP_regression_std, 
                X_training=X, y_training=y
            )
            self.estimator.teach(X, y)

        else:
            n_batches = int((self.n_total - self.batch_size) / self.batch_size)
            n_last_batch = self.n_total - (self.batch_size + n_batches * self.batch_size)

            initial_idx = numpy.random.choice(range(len(X)), size=self.batch_size, replace=False)
            X_training, y_training = X[initial_idx], y[initial_idx]

            self.estimator = ActiveLearner(
                estimator=self.regressor,
                query_strategy=GP_regression_std, 
                X_training=X_training, y_training=y_training
            )

            batches = [self.batch_size for i in range(n_batches)]

            if n_last_batch>0:
                batches.append(n_last_batch)

            for n_queries in batches:
                query_idx, query_instance = self.estimator.query(X, self.batch_size)
                self.estimator.teach(X[query_idx], y[query_idx])

        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError
        return self.estimator.predict(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GPR_AL',
                'name': 'ActiveLearningGaussianProcessRegression',
                'handles_regression': True,
                'handles_classification': False,
                'handles_multiclass': False,
                'handles_multilabel': False,
                'handles_multioutput': False,
                'is_deterministic': False,
                'input': (SPARSE, DENSE, UNSIGNED_DATA, SIGNED_DATA),
                'output': (PREDICTIONS,)}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()

        batch_size = UniformIntegerHyperparameter(
            name="batch_size", lower=10, upper=1000, default_value=50, log=False)
        n_total = UniformIntegerHyperparameter(
            name="n_total", lower=100, upper=1000, default_value=100, log=False)
        alpha = UniformFloatHyperparameter(
            name="alpha", lower=1e-14, upper=1.0, default_value=1e-8, log=True)
        thetaL = UniformFloatHyperparameter(
            name="thetaL", lower=1e-10, upper=1e-3, default_value=1e-6, log=True)
        thetaU = UniformFloatHyperparameter(
            name="thetaU", lower=1.0, upper=100000, default_value=100000.0, log=True)
        
        
        cs.add_hyperparameters([batch_size, n_total, alpha, thetaL, thetaU])
        return cs


# def grid_nn():
#     shapes = ["funnel", "brick"]
#     neuron_max = [1, 2, 3, 4]
#     n_layers = [1, 2, 3]

#     models = []
#     for shape in shapes:
#         for neurons in neuron_max:
#             for n in n_layers:
#                 models.append( \
#                     {
#                         "model_type":ShapedNeuralNetwork,
#                         "hyperparams":
#                         {
#                             "n_inputs":None,
#                             "n_layers":n, 
#                             "neuron_max": neurons, 
#                             "activation":"relu", 
#                             "batch_norm":True, 
#                             "shape":shape, 
#                             "batch_size":64, 
#                             "epochs":100, 
#                             "validation_split":0.0
#                         }
#                    },
#                 )

#     return models

def initialize_model(model_type, hyperparams):

    if model_type == AutoSklearnRegressor:
        if "ActiveLearningGaussianProcessRegression" in hyperparams["include_estimators"]:
            autosklearn.pipeline.components.regression.add_regressor(ActiveLearningGaussianProcessRegression)

    model = model_type(**hyperparams)

    return model


def get_models(model_setup, config=config):
    """
    Function to define which models to use in experiments.
    Uses a hardcoded dictionary to create several different configurations.
    TODO: use config file to create model configurations.

    Config:
        model_setup (str): Name of model configuration to return.

    Returns: 
        models[model_setup]: Return a list of dicts describing models to be used in experiment.

    """

    models = {}

    default_regressors = ["adaboost", "ard_regression", "decision_tree", \
    "extra_trees", "gaussian_process", "gradient_boosting", "k_nearest_neighbors", \
    "liblinear_svr", "libsvm_svr", "mlp", "random_forest", "sgd"]

    models["test"] = \
    [
        # Random Forest
        {
            "model_type":RandomForestRegressor,
            "hyperparams":
            {
                "n_estimators": 1
            }
       },

       {
            "model_type":LinearRegression,
            "hyperparams":
            {
                "fit_intercept": True
            }
       }

    ]

    models["autosklearn-1"] = \
    [
    # Auto-Sklearn, no ensemble
       {
            "model_type":AutoSklearnRegressor,
            "model_name": "Autosklearn-1",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "max_models_on_disc":1,
                "include_estimators":default_regressors
            }
       },    
    ]

    models["autosklearn-5"] = \
    [
    # Auto-Sklearn, ensemble size 5
       {
            "model_type":AutoSklearnRegressor,
            "model_name": "Autosklearn-5",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":5,
                "ensemble_size":5,
                "ensemble_nbest":5,
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "max_models_on_disc":5,
                "include_estimators":default_regressors
            }
       },
    ]

    models["RF"] = \
    [
    # RandomForest
        {
            "model_type":AutoSklearnRegressor,
            "model_name": "RF",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "max_models_on_disc":1,
                "include_estimators":["random_forest"]
            }
       },
    ]

    models["MLP"] = \
    [
    # RandomForest
        {
            "model_type":AutoSklearnRegressor,
            "model_name": "MLP",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "max_models_on_disc":1,
                "include_estimators":["mlp"]
            }
       },
    ]

    models["baseline"] = \
    [
    # RandomForest
        {
            "model_type":AutoSklearnRegressor,
            "model_name": "RF",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "max_models_on_disc":1,
                "include_estimators":["random_forest"]
            }
       },

    # MLP
        {
            "model_type":AutoSklearnRegressor,
            "model_name": "MLP",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "include_estimators":["mlp"],
                "max_models_on_disc":1,
            }
       },
    ]

    models["GPR"] = \
    [
    # GPR + AL
       {
            "model_type":AutoSklearnRegressor,
            "model_name": "GPR",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                # Custom regression estimators do not currently work with multiprocessing (not even the example given by the authors)
                "max_models_on_disc":1,
                "n_jobs":1,
                "delete_tmp_folder_after_terminate":False,
                "include_estimators":["gaussian_process"]
            }
       },

    ]

    models["yield_baseline"] = \
    [
    # MLP
        {
            "model_type":AutoSklearnRegressor,
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "max_models_on_disc":1,
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "include_estimators":["mlp"]
            }
       },
       # GPR + AL
       {
            "model_type":AutoSklearnRegressor,
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":0,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                "max_models_on_disc":1,
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                # Custom regression estimators do not currently work with multiprocessing (not even the example given by the authors)
                "n_jobs":1,
                "delete_tmp_folder_after_terminate":False,
                "include_estimators":["gaussian_process"]
            }
       },
        # Auto-Sklearn, ensemble size 5
       {
            "model_type":AutoSklearnRegressor,
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":5,
                "ensemble_size":5,
                "ensemble_nbest":5,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                "max_models_on_disc":5,
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "include_estimators":default_regressors
            }
       },
    ]

    models["12hr_baseline"] = \
    [
        # RandomForest
        {
            "model_type":AutoSklearnRegressor,
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "max_models_on_disc":1,
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "include_estimators":["random_forest"]
            }
       },

        # MLP
        {
            "model_type":AutoSklearnRegressor,
            "model_name": "MLP",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":1,
                "ensemble_size":1,
                "ensemble_nbest":1,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
                "max_models_on_disc":1,
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "include_estimators":["mlp"]
            }
       },
       # GPR + AL
       # {
       #      "model_type":AutoSklearnRegressor,
       #      "hyperparams":
       #      {
       #          "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
       #          "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
       #          "initial_configurations_via_metalearning":0,
       #          "ensemble_size":1,
       #          "ensemble_nbest":1,
       #          "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
       #          "tmp_folder":uniquify_folder(tmp_folder),
       #          "max_models_on_disc":5,
       #          # "tmp_folder":config["models"]["auto_sklearn"]["tmp_output"],
       #          # Custom regression estimators do not currently work with multiprocessing (not even the example given by the authors)
       #          "n_jobs":1,
       #          "delete_tmp_folder_after_terminate":False,
       #          "include_estimators":["gaussian_process"]
       #      }
       # },
        # Auto-Sklearn, ensemble size 5
       {
            "model_type":AutoSklearnRegressor,
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":5,
                "ensemble_size":5,
                "ensemble_nbest":5,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                "max_models_on_disc":5,
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "include_estimators":default_regressors
            }
       },
    ]

    models["proposed"] = \
    [
        # Auto-Sklearn, ensemble size 5
       {
            "model_type":AutoSklearnRegressor,
            "model_name": "Proposed",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":5,
                "ensemble_size":5,
                "ensemble_nbest":5,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "max_models_on_disc":5,
                "include_estimators":default_regressors,
            }
       },
    ]

    models["proposed_ensemble"] = \
    [
        # Auto-Sklearn, ensemble size 5
       {
            "model_type":AutoSklearnRegressor,
            "model_name": "Proposed ensemble",
            "hyperparams":
            {
                "time_left_for_this_task": config["models"]["auto_sklearn"]["total_time"],
                "per_run_time_limit":config["models"]["auto_sklearn"]["time_per_task"],
                "initial_configurations_via_metalearning":5,
                "ensemble_size":5,
                "ensemble_nbest":5,
                "memory_limit":config["models"]["auto_sklearn"]["memory_limit"],
                "tmp_folder":uniquify_folder(tmp_folder),
                "n_jobs":config["models"]["auto_sklearn"]["n_jobs"],
                "delete_tmp_folder_after_terminate":False,
                "max_models_on_disc":5,
                "include_estimators":default_regressors,
            }
       },
    ]    
   

    return models[model_setup]

