#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 14:57:35 2025

@author: egom802
"""
#%%
#Load csv files
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

# Constants
EARTH_RADIUS_KM = 6371.0

# % Load transects (NZ only)
transects = gpd.read_file("transects_extended.geojson")
transects = transects[transects.site_id.str.startswith("nzd")]

# % Shoreline position for ref year 2005
shoreline_2005_gdf = gpd.read_file("points_ref_shoreline_2005.geojson").to_crs(epsg=2193)

# Drop missing geometries 
shoreline_2005_gdf = shoreline_2005_gdf.dropna(subset=["geometry"])

# Keep only transects that have a shoreline and vice versa
common_ids = set(transects["id"]) & set(shoreline_2005_gdf["transect_id"])

transects = transects[transects["id"].isin(common_ids)].reset_index(drop=True)
shoreline_2005_gdf = shoreline_2005_gdf[shoreline_2005_gdf["transect_id"].isin(common_ids)].reset_index(drop=True)

print(len(transects), len(shoreline_2005_gdf))  # should match

# %% Convert shoreline points -> lat/lon DataFrame
shoreline_latlon = shoreline_2005_gdf.to_crs(epsg=4326)
coastsat = pd.DataFrame({
    "lon": shoreline_latlon.geometry.x,
    "lat": shoreline_latlon.geometry.y
})

# % Sea Level Rise data (NZ SeaRise lat/lon)
df_latlon = pd.read_csv("NZ_VLM_final_May24.csv")
nzrise = df_latlon[["Lon", "Lat"]].rename(columns={"Lon": "lon", "Lat": "lat"})

# %% Nearest neighbor search with BallTree
# Convert to radians
coastsat_rad = np.deg2rad(coastsat[["lat", "lon"]].values)
nzrise_rad = np.deg2rad(nzrise[["lat", "lon"]].values)

# Build BallTree
tree = BallTree(nzrise_rad, metric="haversine")

# Query nearest neighbor
distances, indices = tree.query(coastsat_rad, k=1)
distances_km = distances.flatten() * EARTH_RADIUS_KM
nearest_indices = indices.flatten()
#%%
# %% Get nearest NZRise rows and combine
nearest_nzrise = (
    nzrise.iloc[nearest_indices]
    .reset_index(drop=True)
    .add_prefix("nzrise_")
)

combined = (
    coastsat.reset_index(drop=True)
    .assign(
        distance_km=distances_km,
        site_ID_nzrise=df_latlon["Site ID"].iloc[nearest_indices].values,
        beach_slope=transects.beach_slope.values,
        coastsat_id=transects.id.values,
        trend=transects.trend.values,
    )
)

# Flag cases farther than threshold
threshold = 2
count = (combined["distance_km"] > threshold).sum()
print(f"Points farther than {threshold} km: {count}")


#%%
#Download actual SLR and VLM csv
# df_nzrise_slr= pd.read_csv("https://zenodo.org/records/14722058/files/NZ_Searise_noVLM-2005.csv")

df_nzrise_slr= pd.read_csv('NZ_Searise_noVLM-2005.csv')

#%%
#For each different tag in coastsat_id in df_combined, 
#obtain the subset in df_nzrise_slr that matches the 
#site_ID_nzrise tag in df_combined
#Version without for loop, much faster with large datasets

df_nzrise_slr.rename(columns={'site': 'site_ID_nzrise'},
                                         inplace=True)

# Merge df_combined with df_nzrise_slr using the site_ID_nzrise column
merged_df = combined.merge(
    df_nzrise_slr,
    on='site_ID_nzrise',
    how='left'  # or 'inner' if you only want matching entries
)

#%% Calculate all retreat
#Multiply percentile columns 17, 50, 83

slr_cols = ['17', '50', '83']

# Create a DataFrame of just the SLR columns
slr_df = merged_df[slr_cols]

# Divide all columns in slr_df by beach_slope (row-wise)
#Bruun rule applied
# Shoreface slope multiplied by 0.5 to obtain 
# Bruun beach slope to half of the foreshore beach slope 
# # a bit ad hoc, applied in Vitousek et al. (2023)
#  but is somewhat consistent with Lidar profiles in California

c_adjust = 0.5
retreat_df = slr_df.div(np.tan(merged_df['beach_slope'] * c_adjust), axis=0)

# Rename columns
retreat_df = retreat_df.rename(columns=lambda x: f'retreat_{x}')

# Append to merged_df
merged_df = pd.concat([merged_df, retreat_df], axis=1)

# Add historic rate
# historic_retreat_df = retreat_df.add((merged_df['year'] - 2005)*merged_df['trend'], axis=0 )
# merged_df = pd.concat([merged_df, historic_retreat_df], axis=1)


#%%Export separate files per scenario/year:

import geopandas as gpd
from shapely.geometry import Point

# Convert to geometry
#change this from combined to merged

url_sv_gj="/home/egom802/Documents/GitHub/OCC_Retreat_Modelling/"

# Get unique combinations
unique_years =  [2020]
# unique_years =  [2005, 2020, 2030, 2050, 2080, 2100]
# unique_years =  [2005, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
# unique_scenarios = [1.9,2.6,4.5,7,8.5]
unique_scenarios = [1.9]
#Scenarios 2.6 and 8.5 seem to have duplicates?

# unique_years = merged_df['year'].unique()
# unique_scenarios = merged_df['scenario'].dropna().unique()

# Loop through all unique (SSP, year) combinations
for year in unique_years:
    for scenario in unique_scenarios:
        
        subset = merged_df[(merged_df['year'] == year) & (merged_df['scenario'] == scenario)]
        #Calculate geometry here
        #Drop duplicates, keep the second one, which is the one usually with medium confidence
        subset = subset.drop_duplicates(subset='coastsat_id', keep='last')
        geometry = [Point(xy) for xy in zip(subset['lon'], subset['lat'])]
        
        if not subset.empty:
            # Create save filename
            save_scenario = str(scenario)
            filename = f"retreat_{save_scenario}_{year}_50percentile.geojson"

            #Recalc retreated points from reference, along transects
            subset.retreat_50

            #First, merge subset (retreat dic) with transects and 2005 ref points

            merged_df = pd.merge(subset, transects[['id','geometry']], left_on='coastsat_id',
                                  right_on='id', how='inner')
            merged_df.drop(columns=['id'], inplace=True)

            #Reproject 2005 points to lon lat degrees
            points_2005= shoreline_2005_gdf.to_crs('EPSG:4326')
            points_2005.rename(columns={'geometry': 'points'}, inplace=True)
            points_2005.set_geometry('points', inplace=True)
            points_2005.crs

            lol= pd.merge(merged_df,points_2005[['site_id','transect_id','points']], left_on='coastsat_id',
                                  right_on='transect_id', how='inner')
            lol.drop(columns=['transect_id'], inplace=True)

            lol.rename(columns={"coastsat_id": "id"}, inplace=True)

            ref_points = points_2005
            ref_points.rename(columns={"transect_id":"id"},inplace=True)


#%%
            #Intersect
            # Rename column so it matches transect IDs
            site_id = "nzd0001"
            
            # Get all transects for the site
            site = transects[transects.site_id == site_id]
            site.set_index("id", inplace=True)
            site
            # Filter retreat distances for that site
            distance = lol[lol.site_id == site_id]
            distance.set_index("id", inplace=True)

            ref_points = ref_points[ref_points.site_id == site_id]
            ref_points.set_index("id", inplace=True)

            # Generate interpolated points
            points_2100 = []
            for transect_id, transect in site.iterrows():
                if transect_id not in distance.index:
                    print(f"⚠️ Retreat value not found for transect {transect_id}")
                    continue

                if transect_id not in ref_points.index:
                    print(f"⚠️ Reference point not found for transect {transect_id}")
                    continue

                retreat_distance = distance.loc[transect_id, "retreat_50"]
                ref_point = ref_points.loc[transect_id, "points"]
                transect_line = transect.geometry  # This is a LineString

                # Project reference point onto the transect line (get distance along the line)
                ref_distance_along_line = transect_line.project(ref_point)

                # Calculate new distance from start of line
                new_distance = ref_distance_along_line - retreat_distance

                # Interpolate point at new distance
                new_point = transect_line.interpolate(new_distance)
                points_2100.append(new_point)


#%%Plot
# 1. Create GeoDataFrame for 2100 points
gdf_2100 = gpd.GeoDataFrame(geometry=points_2100, crs=ref_points.crs)

# 2. Reproject everything to EPSG:2193 (NZTM2000)
ref_points_nztm = ref_points.to_crs(epsg=2193)
gdf_2100_nztm = gdf_2100.to_crs(epsg=2193)

# Add basemap (Web Mercator reprojection)
gdf_2100_web = gdf_2100_nztm.to_crs(epsg=3857)
ref_points_web = ref_points_nztm.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))
ref_points_web.plot(ax=ax, color='blue', markersize=20, label='Reference Points')
gdf_2100_web.plot(ax=ax, color='red', markersize=20, label='Projected 2100 Points')
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)  # or another basemap


ax.legend()
ax.set_title("NZ Shoreline Retreat Projection - 2100", fontsize=14)
ax.set_axis_off()
plt.tight_layout()
plt.show()


# %%
