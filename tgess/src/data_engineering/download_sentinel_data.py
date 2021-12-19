import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from time import sleep, strftime
from datetime import date, timedelta, datetime, timezone
import ee
tqdm.pandas()

from .helper_functions import *

# Read config file
config = read_config()

def get_s2_cloud_collection(aoi, date_start, date_end):
    # From: https://developers.google.com/earth-engine/tutorials/community/sentinel-2-s2cloudless
    # Import and filter S2 SR.
    s2_col = (ee.ImageCollection('COPERNICUS/S2')
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        )

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(date_start, date_end))

    # Join the filtered s2cloudless collection to the S2 collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def add_cloud_band(img):
    # Get s2cloudless image, subset the probability band.
    cld_prb = ee.Image(img.get('s2cloudless')).select('probability').rename('cloud_probability')

    # Add the cloud probability layer and cloud mask as image bands.
    return img.addBands(ee.Image([cld_prb]))


def reflectance_time_series(df, path="../data/processed/", filename="GBOV_toa_reflectance_time_series_per_plot.csv"):
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
    
    # For each unique plotID
    output = []
    
    unique_plots = df["plotID"].unique()
    
    missing_counts = {plotID:0 for plotID in unique_plots}
    
    for plotID in tqdm(unique_plots):
        sub_df = df[df["plotID"]==plotID]
                
        # Get first and last date, buffer 1 month around date
        date_start = datetime.strptime(sub_df["date"].min(), '%Y-%m-%d') - timedelta(days=30)
        date_end = datetime.strptime(sub_df["date"].max(), '%Y-%m-%d') + timedelta(days=30)

        # Get location
        aoi = ee.Geometry.Point(sub_df.iloc[0]["longitude"], sub_df.iloc[0]["latitude"])
        
        # Retrieve time series from GEE
        collection = get_s2_cloud_collection(aoi, date_start, date_end).map(add_cloud_band)
        
        count = collection.size()
        
        if count.getInfo()>0:           
            # Retrieve properties as time series (list)
            zenith_angle = collection.aggregate_array("MEAN_SOLAR_ZENITH_ANGLE").getInfo()
            azimuth_angle = collection.aggregate_array("MEAN_SOLAR_AZIMUTH_ANGLE").getInfo()
                       
            # Retrieve pixel values for region as time series (list of lists)
            values = collection.getRegion(geometry=aoi, scale=10)
            values = values.getInfo()
                        
            # Create dataframe
            values = pd.DataFrame(values)
            
            # First row is header
            values.columns = values.iloc[0]
            values = values[1:]
            
            # Create columns for properties time series
            values["zenith_angle"]=zenith_angle
            values["azimuth_angle"]=azimuth_angle
            values["plotID"] = plotID
            values["retrieval_datetime"]=pd.to_datetime(values["time"], unit='ms')
                        
            # Retrieve solar irradiance and observer angle per band as time series (list)
            for band in bands:
                # Solar irradiance
                values["solar_irradiance_{}".format(band)] = \
                collection.aggregate_array("SOLAR_IRRADIANCE_{}".format(band)).getInfo()
                
                # Incidence azimuth
                values["incidence_azimuth_{}".format(band)] = \
                collection.aggregate_array("MEAN_INCIDENCE_AZIMUTH_ANGLE_{}".format(band)).getInfo()
                
                # Incidence zenith
                values["incidence_zenith_{}".format(band)] = \
                collection.aggregate_array("MEAN_INCIDENCE_ZENITH_ANGLE_{}".format(band)).getInfo()
                
            values["mean_incidence_azimuth"] = values[["incidence_azimuth_{}".format(band) for band in bands]].mean(axis=1)
            values["mean_incidence_zenith"] = values[["incidence_zenith_{}".format(band) for band in bands]].mean(axis=1)

            output.append(values)   
        
        else:
            print("Missing data for {} in range {} to {}".format(plotID, date_start, date_end))
            missing_counts[plotID]+=1

            
    output = pd.concat(output)
    output.to_csv((path+filename))
    # print("Saved results as {}{}".format(path, filename))

    return output

def nearest(df, date):
    # From: https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date/32237949
    try:
        result = df.set_index("retrieval_datetime").sort_index().index.get_loc(date, method='nearest')
        return result

    except Exception as e:
        print(e)
        print(df)
        return None

def get_closest_cloud_free_data(x, ts_df):
    cols_to_drop = ['id', 'longitude', 'latitude', 'time', 'plotID']
    date = datetime.strptime(x["date"], "%Y-%m-%d")
    subset = ts_df[ts_df["plotID"]==x["plotID"]]
    subset = subset.drop(columns=cols_to_drop)
    cols = subset.columns
    nearest_idx = nearest(subset, date)

    if nearest_idx is None:
        # Concatenate None values
        x = pd.concat([x, pd.Series({col:None for col in cols})])
        x["abs_date_difference"] = None
        
    else:
        # Concatenate reflectance values from nearest date
        x = pd.concat([x, subset.iloc[nearest_idx]])
        x["abs_date_difference"] = abs(x["datetime"]-x["retrieval_datetime"]).days

    return x

def reflectance_series_to_dataframe(df, ts_df, max_cloud_prb=70, max_date_diff=14, path="../data/processed/", filename="GBOV_LAI_toa_reflectances.csv"):
  
    # Drop duplicate rows, somehow does not work with float or datetime columns (rounding errors?)
    # But does work with date
    ts_df["retrieval_date"]=ts_df["retrieval_datetime"].apply(lambda x: x.date())

    ts_df = ts_df.drop_duplicates(subset=["retrieval_datetime", "latitude", "longitude"], keep="first")

    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    ts_df["retrieval_datetime"] = pd.to_datetime(ts_df["retrieval_datetime"]).dt.tz_localize(None)   
    ts_df = ts_df[ts_df["cloud_probability"]<max_cloud_prb]
    
    df = df.progress_apply(get_closest_cloud_free_data, args=(ts_df,), axis=1)
    
    n_unfiltered = len(df)
    df.to_csv(filename[:-4]+"_unfiltered.csv")
    
    df = df[~df["retrieval_datetime"].isna()]

    n_dropna = len(df)
    df.to_csv(filename[:-4]+"_only_valid_measurements.csv")
    print("Lost {} samples from dropping NaN measurements.".format(n_unfiltered-n_dropna))
    
    df = df[df["abs_date_difference"]<=max_date_diff]
    df.to_csv(filename[:-4]+"_filtered.csv")
    n_date_diff = len(df)
    print("Lost {} samples from filtering on date difference.".format(n_dropna-n_date_diff))
    
    return df

def map_to_degrees(x):
    if x<0:
        x = 360+x
    else:
        x = x
    return x

def preprocess_angles(df):
    rename_cols = {"zenith_angle":"solar_zenith", 
                   "azimuth_angle":"solar_azimuth",
                   "mean_incidence_azimuth":"observer_azimuth",
                   "mean_incidence_zenith":"observer_zenith"
                  }
    
    df = df.rename(columns = rename_cols)
        
    df["relative_azimuth"] = (df["solar_azimuth"] - df["observer_azimuth"])\
                                            .apply(map_to_degrees)
    
    df = df.dropna()

    return df

