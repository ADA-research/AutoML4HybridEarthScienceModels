"""
Most functions in this file are adapted from Sam Murphy's code: https://github.com/samsammurphy/gee-atmcorr-S2
"""


import pandas as pd
import numpy as np
import os
import math
import ee
from tqdm.auto import tqdm
from time import sleep, strftime
from datetime import date, timedelta, datetime, timezone
from Py6S import *
from .atmospheric import Atmospheric
from .helper_functions import *

# Read config file
config = read_config()

# Enable progress bars for pandas apply()
tqdm.pandas()


def get_atmosphere_features(x):

    date = ee.Date(str(x["retrieval_date"]))
    aoi = ee.Geometry.Point(x["longitude"], x["latitude"])

    x["atmosphere_water"] = Atmospheric.water(aoi, date).getInfo()
    x["atmosphere_ozone"] = Atmospheric.ozone(aoi, date).getInfo()
    x["atmosphere_aerosol"] = Atmospheric.aerosol(aoi, date).getInfo()
    
    return x


def get_all_atmosphere_features(df, path, filename):  

    df = df.progress_apply(get_atmosphere_features, axis=1)
    
    df.to_csv(path+filename[:-4]+"_atmospheric_features.csv")
    
    return df

def spectralResponseFunction(bandname):
    """
    Extract spectral response function for given band name
    """
    bandSelect = {
        'B1':PredefinedWavelengths.S2A_MSI_01,
        'B2':PredefinedWavelengths.S2A_MSI_02,
        'B3':PredefinedWavelengths.S2A_MSI_03,
        'B4':PredefinedWavelengths.S2A_MSI_04,
        'B5':PredefinedWavelengths.S2A_MSI_05,
        'B6':PredefinedWavelengths.S2A_MSI_06,
        'B7':PredefinedWavelengths.S2A_MSI_07,
        'B8':PredefinedWavelengths.S2A_MSI_08,
        'B8A':PredefinedWavelengths.S2A_MSI_8A,
        'B9':PredefinedWavelengths.S2A_MSI_09,
        'B10':PredefinedWavelengths.S2A_MSI_10,
        'B11':PredefinedWavelengths.S2A_MSI_11,
        'B12':PredefinedWavelengths.S2A_MSI_12,
        }
    return Wavelength(bandSelect[bandname])

def toa_to_rad(bandname, solar_irradiance, toa_reflectance, date, s):
    """
    Converts top of atmosphere reflectance to at-sensor radiance
    """
    
    # solar exoatmospheric spectral irradiance
    solar_angle_correction = math.cos(math.radians(s.geometry.solar_z))
    
    # Earth-Sun distance (from day of year)
    doy = date.timetuple().tm_yday # day of year
    d = 1 - 0.01672 * math.cos(0.9856 * (doy-4))# http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
   
    # conversion factor
    multiplier = solar_irradiance*solar_angle_correction/(math.pi*d**2)

    # at-sensor radiance
    rad = toa_reflectance*multiplier
    
    return rad

def surface_reflectance(bandname, solar_irradiance, toa_reflectance, date, s):
    """
    Calculate surface reflectance from at-sensor radiance given waveband name
    """
    
    # run 6S for this waveband
    s.wavelength = spectralResponseFunction(bandname)
    s.run()
    
    # extract 6S outputs
    Edir = s.outputs.direct_solar_irradiance             #direct solar irradiance
    Edif = s.outputs.diffuse_solar_irradiance            #diffuse solar irradiance
    Lp   = s.outputs.atmospheric_intrinsic_radiance      #path radiance
    absorb  = s.outputs.trans['global_gas'].upward       #absorption transmissivity
    scatter = s.outputs.trans['total_scattering'].upward #scattering transmissivity
    tau2 = absorb*scatter                                #total transmissivity
    
    # radiance to surface reflectance
    rad = toa_to_rad(bandname, solar_irradiance, toa_reflectance, date, s)
    ref = ((rad-Lp)*math.pi)/(tau2*(Edir+Edif))
    
    return ref

def apply_atmospheric_correction(x, bands):
    s = SixS()

    # Atmospheric constituents
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(x["atmosphere_water"],x["atmosphere_ozone"])
    s.aero_profile = AeroProfile.Continental
    s.aot550 = x["atmosphere_aerosol"]

    # Earth-Sun-satellite geometry
    s.geometry = Geometry.User()
    s.geometry.view_z = 0               # always NADIR (I think..)
    s.geometry.solar_z = x["zenith_angle"]        # solar zenith angle
    s.geometry.month = x["retrieval_date"].month # month used for Earth-Sun distance
    s.geometry.day = x["retrieval_date"].day     # day used for Earth-Sun distance
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_custom_altitude(x["Elevation"]) # elevation in km
    
    for band in bands:
        x[band] = surface_reflectance(band, x["solar_irradiance_{}".format(band)], x[band], x["retrieval_date"], s)
        
    del s
    return x

def create_atm_corrected_data(df, path, filename):
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
        
    df["retrieval_date"] = pd.to_datetime(df["retrieval_date"])

    # Correct reflectance values (Sentinel-2 values are upscaled by factor 10000)
    for band in bands:
        df[band] = df[band].apply(lambda x: x/10000)

    # Convert elevation from meters to kilometers
    df["Elevation"] = df["Elevation"].apply(lambda x: x/1000)
    
    df = df.progress_apply(apply_atmospheric_correction, args=(bands,), axis=1)
    
    print("Saving results to file: {}{}.".format(out_path, out_filename))
    df.to_csv(out_path+out_filename)
    
    return df

