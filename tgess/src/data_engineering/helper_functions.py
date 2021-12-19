import yaml
import ee
import os

from pathlib import Path

# Get absolute path
mod_path = (Path(__file__).parent).resolve()

def read_config(config_file="config.yaml"):
    """
    Function to visualize correlation between features and target variable for both the in situ dataset and the simulation dataset.
    
    Args:
        config_file (str): Filepath of yaml configuration file

    Returns:
        config (dict): Parsed configuration file
    """

    # path = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(mod_path, config_file)

    # Load the configuration file
    stream = open(path, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    
    return config


def earth_engine_init():

    # Trigger the authentication flow.
    ee.Authenticate()

    # Initialize the library.
    ee.Initialize()

    return True