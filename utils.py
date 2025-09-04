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
                                 crs_nztm: int,
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
    merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{crs_nztm}", geometry= 'geom_points_ref2005')

    print(f"Merged {len(merged)} transects/shorelines")
    return merged.reset_index(drop=True)
#%%

coastsat_merged = load_and_merge_coastsat_data(
    "transects_extended.geojson",
    "points_ref_shoreline_2005.geojson",
    2193
)

# %%
def load_metadata_nzrise(filepath:str):
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

    return meta_data

def load_slrdata_nzrise(filepath:str):
    """Load SLR data from NZRise project"""
    slr_data = (
        pd.read_csv(filepath)
        .rename(columns={"site": "nzrise_site_id"})
    )
    return slr_data

meta_data_fp = "NZ_VLM_final_May24.csv"
slr_fp = "NZ_Searise_noVLM-2005.csv"
meta_data = load_metadata_nzrise(meta_data_fp)
slr_data  = load_slrdata_nzrise(slr_fp)

nzrise_merged= pd.merge(meta_data,slr_data,
                        on='nzrise_site_id', how='outer')
