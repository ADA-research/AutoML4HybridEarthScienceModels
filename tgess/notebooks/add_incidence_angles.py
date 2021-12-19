import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import IPython.display as disp
from time import sleep, strftime
from datetime import date, timedelta, datetime, timezone
import ee

# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()

tqdm.pandas()

df = pd.read_csv("../data/processed/GBOV_LAI_toa_reflectances_6S_corrected.csv")

def get_incidence_angles(x):
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

    aoi = ee.Geometry.Point(x["longitude"], x["latitude"])
    
    start_date = datetime.strptime(x["retrieval_date"], '%Y-%m-%d')
    end_date = datetime.strptime(x["retrieval_date"], '%Y-%m-%d') + timedelta(days=1)
      
#     collection = get_s2_cloud_collection(aoi, retrieval_date, retrieval_date).map(add_cloud_band)
    img = ee.ImageCollection('COPERNICUS/S2')\
                    .filterBounds(aoi)\
                    .filterDate(ee.Date(start_date), ee.Date(end_date))\
                    .first()

    # Retrieve observer angle per band as time series (list)
    for band in bands:

        # Incidence azimuth
        x["incidence_azimuth_{}".format(band)] = \
        img.get("MEAN_INCIDENCE_AZIMUTH_ANGLE_{}".format(band)).getInfo()

        # Incidence zenith
        x["incidence_zenith_{}".format(band)] = \
        img.get("MEAN_INCIDENCE_ZENITH_ANGLE_{}".format(band)).getInfo()

    x["mean_incidence_azimuth"] = x[["incidence_azimuth_{}".format(band) for band in bands]].mean()
    x["mean_incidence_zenith"] = x[["incidence_zenith_{}".format(band) for band in bands]].mean()

    return x
    
df = df.progress_apply(get_incidence_angles, axis=1)
df.to_csv("../data/processed/GBOV_LAI_angles.csv")