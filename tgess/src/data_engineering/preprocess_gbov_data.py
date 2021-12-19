import numpy as np
import pandas as pd
# import fiona
# import geopandas as gpd
import os
# import geojson
import time
from shapely.geometry import Polygon
# from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date, timedelta, datetime
from tqdm.auto import tqdm

tqdm.pandas()

def compute_LAI(x):
    """
    Computes LAI based on upwards and downwards measurements and filters on quality flags.
    For some sites (i.e. Woodworth), only upwards measurements are available, these are still valid measurements.
    """
        
    # Filter out measurements with high impact issues (flag>=16)
    # See page 22 of https://gbov.acri.fr/public/docs/products/2019-11/GBOV-ATBD-RM4-RM6-RM7_v2.0-Vegetation.pdf
    if x["up_flag"] >=16 or x["down_flag"] >=16:
        x["LAI_Miller"] = np.nan
        x["LAI_Warren"] = np.nan
    
    # LAI_Miller
    if x["LAI_Miller_up"] >= 0 and x["LAI_Miller_down"] >=0:
        x["LAI_Miller"] = x["LAI_Miller_up"] + x["LAI_Miller_down"]
    elif x["LAI_Miller_up"] < 0:
        x["LAI_Miller"] = x["LAI_Miller_down"]  
    elif x["LAI_Miller_down"] < 0:
        x["LAI_Miller"] = x["LAI_Miller_up"]
    else:
        x["LAI_Miller"] = np.nan
    
    # LAI_Warren
    if x["LAI_Warren_up"] >= 0 and x["LAI_Warren_down"] >=0:
        x["LAI_Warren"] = x["LAI_Warren_up"] + x["LAI_Warren_down"]
    elif x["LAI_Warren_up"] < 0:
        x["LAI_Warren"] = x["LAI_Warren_down"]
    elif x["LAI_Warren_down"] < 0:
        x["LAI_Warren"] = x["LAI_Warren_up"]
    else:
        x["LAI_Warren"] = np.nan
    return x

# def get_tiles(df, path, geojson_file, margin=0.001):
#     aoi_path = os.path.join(path, str(df["Site"]+"_"+geojson_file))
    
#     # Create mini polygon (overlap does not work for points)
#     lat = df["Lon_IS"]
#     lon = df["Lat_IS"]
#     lat_list = [lat-margin, lat+margin, lat+margin, lat-margin]
#     lon_list = [lon-margin, lon-margin, lon+margin, lon+margin]

#     # Create Polygon object
#     features = []
#     polygon = geojson.Polygon([list(zip(lon_list, lat_list))])

#     features = [geojson.Feature(geometry=polygon,
#                             properties={})]

#     # Store polygon as geojson file
#     with open(aoi_path, 'w', encoding='utf8') as fp:
#         geojson.dump(geojson.FeatureCollection(features), fp, sort_keys=True, ensure_ascii=False)

        
#     # Get tiles overlapping with polygon
#     print(polygon)
#     overlap = Sentinel2Overlap(aoi_path)
        
#     tiles = overlap.overlap(limit=0.00001)

#     return tiles

# def get_all_tiles(df, path, geojson_file):
#     #Groupby site
#     temp_df = df[["Site", "Lat_IS", "Lon_IS"]].groupby("Site").max().reset_index()
#     site_tile_map = {}
#     for i, row in tqdm(temp_df.iterrows()):
#         print(row)
#         tiles = get_tiles(row, path, geojson_file)
#         print(tiles)
#         site_tile_map[row["Site"]] = str(tiles)
    
#     df["tiles"] = df.apply(lambda x: site_tile_map[x["Site"]], axis=1)
#     return df
        

def aggregate_files_into_df(path):
    """
    Combine all individual .csv files into one pandas Dataframe
    """
    
    # Select all csv files
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        
    # For each file:
    df = []
    for csv_file in tqdm(csv_files, desc="Creating dataframe from .csv files"):
        # Read data and append to list
        filename = os.path.join(path, csv_file)
        temp_df = pd.read_csv(filename, index_col=None, header=0, sep=";")
#         temp_df["tiles"] = str(get_tiles(temp_df.iloc[0], path, "temp.geojson"))
        df.append(temp_df)

    # Convert list of dataframes to dataframe
    df = pd.concat(df, axis=0, ignore_index=True)
    
        
    #Add LAI columns and filter on quality
    df = df.apply(compute_LAI, axis=1)
    
    #Add datetime
    df["datetime"] = pd.to_datetime(df["TIME_IS"])
    

    
    return df

# def get_tiles_from_df(row, kmlns, elems):
#     # Here's our point of interest
#     p = Point(row["Lon_IS"], row["Lat_IS"])

#     # Filter polygon elements using this lambda (anonymous function)
#     # keytree.geometry() makes a GeoJSON-like geometry object from an
#     # element and shape() makes a Shapely object of that.
#     hits = filter(
#         lambda e: shape(keytree.geometry(e.find(".//{%s}Polygon" % kmlns))).contains(p),
#         elems )
#     # hits is a list of lists of polygon objects
    
#     tiles = []
#     for hit in hits:
#         tiles.append((hit.find("{%s}name" % kmlns).text))

#     return str(tiles)

# def find_tiles(df, tiles):
#     # Parse KML file
#     with open(tiles, 'r') as file:
#         data = file.read()
    
#     tree = ElementTree.fromstring(data)
    
#     kmlns = tree.tag.split('}')[0][1:]

#     # Find all Polygon elements anywhere in the doc
#     elems = tree.findall(".//{%s}Placemark" % kmlns)
    
#     # Group df by site
#     temp_df = df[["Site", "Lat_IS", "Lon_IS"]].groupby("Site").max()
    
#     # For each site, get tile from KML
#     temp_df["tiles"] = temp_df.progress_apply(get_tiles_from_df, args=(kmlns, elems), axis=1)
    
#     # Join dataframes, create column tiles in original df
#     df = df.join(temp_df.drop(columns=["Lon_IS", "Lat_IS"]), on="Site", how="left")
    
#     # Create date_start, date_end and first_tile columns
#     df["date"] = df["datetime"].apply(lambda x: x.date())
#     df["date_start"] = df["datetime"].apply(lambda x: x.date() - timedelta(days=5))
#     df["date_end"] = df["datetime"].apply(lambda x: x.date() + timedelta(days=5))
# #     df["first_tile"] = df["tiles"].apply(lambda x:x.split(",")[0][2:-1])
#     df["first_tile"] = df["tiles"].apply(lambda x: ''.join(c for c in x.split(",")[0] if c not in '\'[]'))
#     df["second_tile"] = df["tiles"].apply(lambda x: ''.join(c for c in x.split(",")[1] if c not in '\'[]') if len(x.split(","))>1 else None)

#     return df

# Commented out due to environment issues on Windows

# def get_plot_coordinates(df, shapefile="data/shapefiles/NEON_TOS_Plot_Centroids.shp"):
#     # Read NEON plot shapefile
#     # From: https://www.arcgis.com/home/item.html?id=73e3e0b777d344eca88573ccd21b19e9
#     df_shp = gpd.read_file(shapefile)
    
#     # Filter on plotID where DHP was used
#     df_shp = df_shp[df_shp['appMods'].str.contains('dhp', na = False)] 
    
#     # Merge with df
#     cols = ["plotID", "longitude", "latitude"]
#     df = df.merge(df_shp[cols], how="inner", left_on="PLOT_ID", right_on="plotID")
#     return df

