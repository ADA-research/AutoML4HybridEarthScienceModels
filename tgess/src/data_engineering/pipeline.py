import os
import pandas as pd
import numpy as np

from .atmospheric_correction import *
from .download_sentinel_data import *
from .preprocess_gbov_data import *
from .helper_functions import *

# Read config file
config = read_config()


def gbov_rm07_preprocessing_pipeline():
    """
    Pipeline to create in situ dataset from raw GBOV RM07 (LAI) files.
    """

    path = "data/GBOV_RM07"
    
    in_path = config["filepaths"]["raw"]
    gbov_folder =  config["gbov_data"]["gbov_folder"]
    in_full_path = os.path.join(path, filename)

    # Create in situ dataset
    df = aggregate_files_into_df(in_full_path)

    # Find sentinel tile names (optional --> not used)
    # tiles = config["gbov_data"]["tiles"]
    # df = find_tiles(df, tiles)

    # Get coordinates for each unique plotID
    shapefile = config["gbov_data"]["shapefile"]
    shapefile = os.path.join(in_path, shapefile)
    df = get_plot_coordinates(df, shapefile) 

    # Save results to .csv
    out_path = config["filepaths"]["intermediate"]
    out_filename = config["sentinel_data"]["input_filename"]
    out_full_path = os.path.join(out_path, out_filename)
    df.to_csv(out_full_path)
    
    return df


def sentinel_data_pipeline(path, filename):
    """
    Pipeline to get satellite reflectance data and metadata for measurements included in the GBOV RM07 data.
    Expects the gbov_rm07_preprocessing_pipeline() to have been run first.
    """

    # Initialize earth engine session
    earth_engine_init()

    # Get reflectance time series for each location in in situ dataset
    path = config["filepaths"]["intermediate"]
    filename =  config["sentinel_data"]["input_filename"]
    full_path = os.path.join(path, filename)

    if os.path.exists(full_path):
        print("File {} already exists, skipping step.".format(full_path))
        ts_df = pd.read_csv(full_path)
    else:
        ts_df = reflectance_time_series(df)

    # Create final satellite dataset
    path = config["atmospheric_correction"]["path"]
    filename = config["atmospheric_correction"]["output_filename"]
    full_path = os.path.join(path, filename)

    if os.path.exists(full_path):
        print("File {} already exists, skipping step.".format(full_path))
        df = pd.read_csv(full_path)
    else:
        max_cloud_prb = int(config["sentinel_data"]["max_cloud_prb"])
        max_date_diff = int(config["sentinel_data"]["max_date_diff"])

        # Choose cloudfree pixels on dates close to in situ sampling dates from whole time series
        df = reflectance_series_to_dataframe(df, ts_df, max_cloud_prb, max_date_diff, path, filename)

        # Rename some columns and compute angles required for prosail
        df = preprocess_angles(df)

    return df


def atmospheric_correction_pipeline():
    """
    Pipeline to atmospherically correct Sentinel2 data using Py6S. 
    Expects the sentinel_data_pipeline() to have been run first.
    """

    # Initialize earth engine session
    earth_engine_init()

    # Get atmospheric constituents data for each row of satellite data
    in_path = config["filepaths"]["intermediate"]
    in_filename = config["sentinel_data"]["output_filename"]
    in_full_path = os.join(in_path, in_filename)

    out_path = config["filepaths"]["intermediate"]
    out_filename = config["atmospheric_correction"]["output_filename"]
    out_full_path = os.join(out_path, out_filename)

    if os.path.exists(out_full_path):
        print("File {} already exists, skipping step.".format(out_full_path))
        df = pd.read_csv(out_full_path)
    else:
        df = pd.read_csv(in_full_path)
        df = get_all_atmosphere_features(df, out_path, out_filename)

    # Apply Py6S
    out_path = config["atmospheric_correction"]["path"]
    out_filename = config["atmospheric_correction"]["output_filename"]
    out_full_path = os.path.join(out_path, out_filename)

    if os.path.exists(out_full_path):
        print("File {} already exists, skipping step.".format(out_full_path))
        df = pd.read_csv(out_full_path)
    else:
        df = create_atm_corrected_data(df, out_path, out_filename)

    return df
