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

def load_points(filepath: str):
    """Load ref points (shoreline) and reproject, drop missing geometries.
    Points externally generated in gen_2005ref.py"""
    points = gpd.read_file(filepath)
    return points.dropna(subset=["geometry"])

def rename_columns(transects: gpd.GeoDataFrame,
                   points: gpd.GeoDataFrame):
    """Rename columns for consistency."""
    transects = transects.rename(columns={
        "site_id": "coastsat_site_id",
        "id": "coastsat_transect_id",
        "geometry": "geom_transect_coastsat"
    })
    points = points.rename(columns={
        "site_id": "coastsat_site_id",
        "transect_id": "coastsat_transect_id",        
        "geometry": "geom_points_ref2005"
    })
    return transects, points

def load_and_merge_coastsat_data(transects_fp: str,
                                 points_fp: str,
                                 crs_str: int,
                                 col_merge_label: str = "coastsat_transect_id"):
    """Wrapper: load, clean, and merge coastsat data."""
    transects = load_transects(transects_fp)
    points = load_points(points_fp)
    transects, points = rename_columns(transects, points)

    # Merge on a column
    merged = pd.merge(transects,
                      points[[
                          c for c in points.columns if c not in transects.columns or c == col_merge_label]],
                      on= col_merge_label,  how="inner")
    merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{crs_str}", geometry= 'geom_points_ref2005')

    print(f"Merged {len(merged)} transects/points")
    return merged.reset_index(drop=True)
# %
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

#%
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

    denom = c_adjust * all_merged["beach_slope"]
    # Apply Bruun rule (beach slope * c_adjust)
    retreat_df = (
        all_merged[slr_quantiles]
        .div(denom , axis=0)
        .rename(columns=lambda c: f"retreat_{c}")
        .round(2)  # ensures 3 decimals
    )

    # Append retreat columns
    merged_retreat_df = pd.concat([all_merged, retreat_df], axis=1)

     #(Optional) Historic rate adjustment. Trend is in (m/year)
    # historic_retreat_df = retreat_df.add(
    #     (all_merged["year"] - 2005) * all_merged["trend"].round(2), axis=0
    # )
    # merged_retreat_df = pd.concat([all_merged, historic_retreat_df], axis=1)

    return merged_retreat_df

def points_to_polylines(subset):
    """ Take the points GeoDataFrame and transforms to polylines"""
    missing = subset[subset.geometry.isnull()]
    print(f"Missing geometries: {len(missing)}")

    subset = subset.dropna(subset=['geom_new_points'])

    # Step 1: Extract group and order fields
    subset["group_id"] = subset["coastsat_transect_id"].str.split("-").str[0]
    subset["order_id"] = subset["coastsat_transect_id"].str.split("-").str[1].astype(int)

    # Step 2: Create LineStrings per group
    lines = []

    for group_id, group in subset.groupby("group_id"):
        sorted_group = group.sort_values(by="order_id")
        coords = sorted_group.geometry.tolist()
        
        # Ensure we have at least 2 points to make a line
        if len(coords) >= 2:
            line = LineString(coords)
            lines.append({"geometry": line, "group_id": group_id})

    # Step 3: Create GeoDataFrame of LineStrings
    lines_gdf = gpd.GeoDataFrame(lines, crs=subset.crs)

    # Step 4: Reproject to WGS84 (EPSG:4326) for web tools like Leaflet / Google Earth
    lines_gdf = lines_gdf.to_crs(epsg=4326)
    return(lines_gdf)

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
    "transects_reindexed.geojson",
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
#%%
#MWE of retreat polyline in one site
# Get unique combinations
years =  [2005]
scenarios = [1.9]
# years =  [2005, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
# unique_scenarios = [1.9,2.6,4.5,7,8.5]
slr_qt =  "50" #quantiles 17,50,83

for year in years:
    for scenario in scenarios:

        # Subset dataframe with specific projection year & specific scenario
        subset = retreat[(retreat['year'] == year) & (retreat['scenario'] == scenario)]
        #subset = subset.drop_duplicates(subset='coastsat_transect_id', keep='last')

        #Calc new point location according to retreat_50
        bruun_slr_qt= subset[f"retreat_{slr_qt}"]             #projected retreat 50 quantile

        subset.geom_transect_coastsat.to_crs(2193).length   #transects lengths
        # Reproject to NZTM2000 (meters)

        # Project reference point onto the transect line (get distance along the line)
        ref_distances_along_lines = subset.geom_transect_coastsat.to_crs(2193) .project(
                                    subset.geom_points_ref2005.to_crs(2193)  )

        # Calculate new distance from start of line
        new_distances = ref_distances_along_lines - bruun_slr_qt
        
        # Interpolate points at new distances
        new_points = subset.geom_transect_coastsat.to_crs(2193).interpolate(new_distances)

        #Because the transformation from transects to points needs cleaning (labels)
        #Here some cleaning is added or even better... clean coastsat transects directly

        #append new_points 
        # subset['geom_new_points'] = subset.geom_transect_coastsat.interpolate(new_distances)
        subset['geom_new_points'] = new_points.to_crs(4326)

        #% From points to linestrings, careful on what active geometry goes inside
        lines_gdf = points_to_polylines(subset.set_geometry('geom_new_points'))
        lines_gdf.to_file(f"lines_shoreline_{slr_qt}qtl_{year}_{scenario}.geojson")
        cols_to_display= ['geom_new_points','50','retreat_50']
        subset[cols_to_display].to_file(f"points_shoreline_{slr_qt}qtl_{year}_{scenario}.geojson")
       
#%%
# Plot new points, compare to ref

# Add basemap (Web Mercator reprojection)
# gdf_2100_web = new_points.to_crs(epsg=3857)
gdf_2100_web = subset.geom_new_points.to_crs(epsg=3857)
ref_points_web = subset.geom_points_ref2005.to_crs(epsg=3857)
transects_web= subset.geom_transect_coastsat.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))
transects_web.plot(ax=ax, color='yellow',  label='Coastsat Transect', zorder=1)
ref_points_web.plot(ax=ax, color='blue', markersize=20, label='Reference Points')
gdf_2100_web.plot(ax=ax, color='red', markersize=20, label= f"Projected :{year} Points")
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)  # or another basemap


ax.legend()
ax.set_title(f"NZ Shoreline Retreat Projection - {year}", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
plt.show()


# %%
