import numpy as np
import pandas as pd
import os
import math
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from tqdm.auto import tqdm
from autosklearn.regression import AutoSklearnRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit
import time
from sklearn.metrics import r2_score
from modAL.models import ActiveLearner


tqdm.pandas()

def main():
    train_df = pd.read_csv("../../data/processed/PROSAIL_LUT_S2_10000_generic.csv")
    validation_df = pd.read_csv("../../data/processed/GBOV_RM07_in_situ_train.csv")
    test_df = pd.read_csv("../../data/processed/GBOV_RM07_in_situ_holdout.csv")

    train_df = train_df.rename(columns={"lai":"LAI_Warren"})

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

    class ActiveLearningGaussianProcessRegression(AutoSklearnRegressionAlgorithm):
        # Based on: https://github.com/automl/auto-sklearn/blob/master/autosklearn/pipeline/components/regression/gaussian_process.py
        
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
            batch_size = UniformIntegerHyperparameter(
                name="batch_size", lower=10, upper=50, default_value=50, log=False)
            n_total = UniformIntegerHyperparameter(
                name="n_total", lower=100, upper=200, default_value=100, log=False)
            alpha = UniformFloatHyperparameter(
                name="alpha", lower=1e-14, upper=1.0, default_value=1e-8, log=True)
            thetaL = UniformFloatHyperparameter(
                name="thetaL", lower=1e-10, upper=1e-3, default_value=1e-6, log=True)
            thetaU = UniformFloatHyperparameter(
                name="thetaU", lower=1.0, upper=100000, default_value=100000.0, log=True)
            
            cs = ConfigurationSpace()
            cs.add_hyperparameters([batch_size, n_total, alpha, thetaL, thetaU])
            return cs

    autosklearn.pipeline.components.regression.add_regressor(ActiveLearningGaussianProcessRegression)

    model = {
                "model_type":AutoSklearnRegressor,
                "hyperparams":
                {
                    "time_left_for_this_task": 1*60,
                    "per_run_time_limit":1*30,
                    "initial_configurations_via_metalearning":0,
                    "ensemble_size":1,
                    "ensemble_nbest":1,
#                     "output_folder":"/tmp/autosklearn_output",
                    "n_jobs":1,
                    "delete_tmp_folder_after_terminate":False,
#                     "delete_output_folder_after_terminate":True,
                    "include_preprocessors":["no_preprocessing"], 
                    "exclude_preprocessors":None,
                    "memory_limit":1024*4,
                    "include_estimators":["ActiveLearningGaussianProcessRegression"]
                }
           }

    target_cols=("lai", "LAI_Warren")
    feature_cols=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', \
                  'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', \
                  'solar_zenith', 'observer_zenith', 'relative_azimuth']
    standardize=False


    model["hyperparams"]["resampling_strategy"] = PredefinedSplit(test_fold=range(len(train_df), len(train_df)+len(validation_df)))

    all_cols = feature_cols.copy()
    all_cols.append(target_cols[1])

    train_data = (train_df[all_cols].append(validation_df[all_cols])).reindex()
    test_data = test_df

    # Train on entire training set
    X_train = np.array(train_data[feature_cols])
    y_train = np.array(train_data[target_cols[1]]) 

    # Shuffle training set
    # idx = np.arange(len(X_train))
    # np.random.shuffle(idx)

    # X_train = X_train[idx]
    # y_train = y_train[idx]

    X_test = np.array(test_data[feature_cols])
    y_test = np.array(test_data[target_cols[1]]) 

    # Standardize data
    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Fit model on entire training set
    t_start = time.process_time()

    # Random Forest
    # model = {
    #     "model_type":RandomForestRegressor,
    #     "hyperparams":
    #     {
    #         "n_estimators": 100,
    #     }
    # }


    model = model["model_type"](**model["hyperparams"])

    print(model)
    model.fit(X_train, y_train)
    model.refit(X_train[:len(train_df)], y_train[:len(train_df)])
    # Get training set predictions  
    y_train_pred = model.predict(X_train)

    train_time = time.process_time()-t_start

    # Get test set predictions
    t_start = time.process_time()
    y_test_pred = model.predict(X_test)
    test_time = time.process_time()-t_start

    print(train_time)
    print(test_time)
    print(r2_score(y_test, y_test_pred))

    m = model.show_models()
    print(model.show_models())
    print(model.sprint_statistics())

if __name__ == "__main__":
    main()