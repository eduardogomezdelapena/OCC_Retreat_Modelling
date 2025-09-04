#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep  4 2025

@author: eduardo.gomez
"""
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
import math
from shapely.geometry import Point, LineString, Polygon
from shapely import line_interpolate_point, get_point
import contextily as ctx
from tqdm import tqdm  # progress bar
from shapely.errors import GEOSException
import matplotlib.pyplot as plt
#%%
# Constants
EARTH_RADIUS_KM = 6371.0
CRS_NZTM = 2193  # NZ Transverse Mercator
CRS_WGS84 = 4326 # Lat/Lon
#%%
def load_transects(filepath: str):
    """Load and filter transects (NZ only)."""
    transects = gpd.read_file(filepath)
    transects = transects.drop_duplicates(subset="id") #drop duplicated transects
    return transects[transects["site_id"].str.startswith("nzd")]

def load_shoreline(filepath: str):
    """Load shoreline and reproject, drop missing geometries."""
    shoreline = gpd.read_file(filepath)
    return shoreline.dropna(subset=["geometry"])

def rename_columns(transects: gpd.GeoDataFrame,
                   shoreline: gpd.GeoDataFrame):
    """Rename columns for consistency."""
    transects = transects.rename(columns={
        "site_id": "coastsat_site_id",
        "id": "coastsat_transect_id",
        "geometry": "geom_transect_coastsat"
    })
    shoreline = shoreline.rename(columns={
        "site_id": "coastsat_site_id",
        "transect_id": "coastsat_transect_id",        
        "geometry": "geom_points_ref2005"
    })
    return transects, shoreline

def load_and_merge_coastsat_data(transects_fp: str,
                                 shoreline_fp: str,
                                 crs_str: int,
                                 col_merge_label: str = "coastsat_transect_id"):
    """Wrapper: load, clean, and merge coastsat data."""
    transects = load_transects(transects_fp)
    shoreline = load_shoreline(shoreline_fp)
    transects, shoreline = rename_columns(transects, shoreline)

    # Merge on a column
    merged = pd.merge(transects,
                      shoreline[[
                          c for c in shoreline.columns if c not in transects.columns or c == col_merge_label]],
                      on= col_merge_label,  how="inner")
    merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{crs_str}", geometry= 'geom_points_ref2005')

    print(f"Merged {len(merged)} transects/shorelines")
    return merged.reset_index(drop=True)
# %%
def load_metadata_nzrise(filepath:str, crs_str: int):
    """Load metadata for NZRise. Return GeoDataFrame"""
    df = (
        pd.read_csv(filepath)
        .rename(columns={"Site ID": "nzrise_site_id"})
    )
    # Drop entirely empty columns
    df = df.dropna(axis=1, how="all")
    # Create geometry column from latitude and longitude
    geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
    meta_data = gpd.GeoDataFrame(df, geometry=geometry)
    meta_data.drop(columns=["Lon","Lat"], inplace=True)
    meta_data.rename_geometry("geom_nzrise",inplace=True)
    meta_data.set_crs(epsg=crs_str,inplace=True)
    return meta_data

def load_slrdata_nzrise(filepath:str):
    """Load SLR data from NZRise project. Returns pd.DataFrame"""
    slr_data = (
        pd.read_csv(filepath)
        .rename(columns={"site": "nzrise_site_id"})
    )
    return slr_data

#%%
def nearest_points(meta_data, coastsat_merged):
    """ Find nearest NZRise points to CoastSat data"""

    #  Nearest neighbor search with BallTree
    # Extract lon/lat from geometries
    coastsat_coords   = np.column_stack((coastsat_merged.geometry.y, coastsat_merged.geometry.x))    
    nzrise_coords = np.column_stack((meta_data.geometry.y, meta_data.geometry.x))

    # Convert to radians
    coastsat_rad = np.deg2rad(coastsat_coords)
    nzrise_rad   = np.deg2rad(nzrise_coords)

    # Build BallTree
    tree = BallTree(nzrise_rad, metric="haversine")

    # Query nearest neighbor
    distances, indices = tree.query(coastsat_rad, k=1)
    distances_km = distances.flatten() * EARTH_RADIUS_KM
    nearest_indices = indices.flatten()

    # Flag cases farther than 2 km 
    threshold = 2; count = (distances_km > threshold).sum()
    print(f"Points farther than {threshold} km: {count}")

    return distances, nearest_indices

def calc_retreat(all_merged, c_adjust = 0.5):
    """ Shoreline retreat calculation. Bruun rule"""

    slr_quantiles = ["17", "50", "83"]

    # c_adjust = 0.5 to adjust the Bruun profile with the shoreface profile
    # a bit ad hoc, matches with some lidar measurements

    # Apply Bruun rule (beach slope * c_adjust)
    retreat_df = (
        all_merged[slr_quantiles]
        .div(np.tan(all_merged["beach_slope"] * c_adjust), axis=0)
        .rename(columns=lambda c: f"retreat_{c}")
    )

    # Append retreat columns
    merged_retreat_df = pd.concat([all_merged, retreat_df], axis=1)

    #  (Optional) Historic rate adjustment
    # historic_retreat_df = retreat_df.add(
    #     (merged_df["year"] - 2005) * merged_df["trend"], axis=0
    # )
    # merged_retreat_df = pd.concat([merged_df, historic_retreat_df], axis=1)

    return merged_retreat_df

#%%

#CRS_NZTM , CRS_WGS84
meta_data_fp = "NZ_VLM_final_May24.csv"
slr_fp = "NZ_Searise_noVLM-2005.csv"
meta_data = load_metadata_nzrise(meta_data_fp, crs_str= CRS_WGS84 ) # gpd.DataFrame
slr_data  = load_slrdata_nzrise(slr_fp) #pd.DataFrame

merged= pd.merge(meta_data,slr_data,
                        on='nzrise_site_id', how='outer')
nzrise_merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{CRS_WGS84}", geometry= 'geom_nzrise')

coastsat_merged = load_and_merge_coastsat_data(
    "transects_extended.geojson",
    "points_ref_shoreline_2005.geojson",
    CRS_WGS84
)

#Calculate distances to nearest NZRise points
distances, nearest_indices = nearest_points(meta_data,
                                             coastsat_merged)

#Merge nzrise_merged & coastsat_merged, based on nearest_indices
#to each coastsat_transect_id there is a matching nzrise_site_id
#nzrise_site_id is repeated, repeat also coastsat_transect_id as many times needed
#add column of nzrise_site_id to coastsat_merged

coastsat_merged["nzrise_site_id"]=meta_data["nzrise_site_id"].iloc[nearest_indices].values

#Now merge based on nzrise_site_id
all_merged = pd.merge( coastsat_merged, nzrise_merged,
                      on='nzrise_site_id', how='outer')
#Print all columns, there should be 3 geometry columns (points ref 2005, coastsat transects
# and nzrise points).

print(all_merged.columns)
#Transform into geopandas? A geometry column needs to be picked, points ref 2005.

all_merged = gpd.GeoDataFrame(all_merged,crs=f"EPSG:{CRS_WGS84}",
                               geometry= 'geom_points_ref2005')

#Now calc retreat
retreat = calc_retreat(all_merged)

#MWE of retreat polyline in one site

#%% Plot all transects and ref points in map

CRS_WEB_MERCATOR = "EPSG:3857"

# Convert data to Web Mercator (EPSG:3857) for plotting with basemap
nzrise_web_mercator = meta_data.to_crs(CRS_WEB_MERCATOR)
coastsat_web_mercator = coastsat_merged.to_crs(CRS_WEB_MERCATOR)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot shoreline and transects in the correct projection
nzrise_web_mercator.plot(ax=ax, color='red', markersize=5, label="NZRise Points")
coastsat_web_mercator.plot(ax=ax, color='blue', markersize=0.5, alpha=0.3, label="Coastsat Points")

# Add the basemap
ctx.add_basemap(ax, crs=CRS_WEB_MERCATOR, source=ctx.providers.Esri.WorldImagery)

# Customize the plot
ax.set_title("2005 Shoreline Points Across All NZ Sites")
ax.legend()
ax.set_axis_off()

# Show the map
plt.show()
# %%
