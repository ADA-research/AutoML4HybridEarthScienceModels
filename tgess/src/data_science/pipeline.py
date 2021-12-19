from .experiment import *
from .models import *
from .helper_functions import *

# Read config file
config = read_config()

# Get root folder of project
root_path = (Path(__file__).parent / ".." / ".." ).resolve()

def run_GBOV_simulation_only_experiment():
    """
    Train several models on PROSAIL LUT and validate on GBOV in situ data.
    Uses config.yaml to configure data filenames, logging and some model specific parameters.
    Models are configured in models.py, in the function get_models().
    Feature and target columns are hard-coded in this function. (TODO: add features to config.yaml)
    
    Returns:
        results (pd DataFrame): 
        plot_data (pd DataFrame):
    """

    models = get_models()

    simulation_target_col = config["experiment"]["data"]["simulation_target_col"]
    in_situ_target_col = config["experiment"]["data"]["in_situ_target_col"]

    feature_cols = config["experiment"]["data"]["feature_cols"]
    target_cols = (simulation_target_col, in_situ_target_col)

    results, plot_data = simulation_only_experiment(models, target_cols, feature_cols)

    return results, plot_data

def run_GBOV_in_situ_only_experiment():
    """
    Train several models on PROSAIL LUT and validate on GBOV in situ data.
    Uses config.yaml to configure data filenames, logging and some model specific parameters.
    Models are configured in models.py, in the function get_models().
    Feature and target columns are hard-coded in this function. (TODO: add features to config.yaml)
    
    Returns:
        results (pd DataFrame): 
        plot_data (pd DataFrame):
    """

    models = get_models()

    in_situ_target_col = config["experiment"]["data"]["in_situ_target_col"]
    
    feature_cols = config["experiment"]["data"]["feature_cols"]
    target_cols = (in_situ_target_col, in_situ_target_col)

    results, plot_data = in_situ_only_experiment(models, target_cols, feature_cols)

    return results, plot_data



if __name__ == "__main__":
    run_GBOV_simulation_only_experiment()