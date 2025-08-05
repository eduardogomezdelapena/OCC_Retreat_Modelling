#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 14:57:35 2025

@author: egom802
"""

#Load csv files
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import LineString, get_point
from sklearn.neighbors import BallTree

#%% #Coastsat data
file_path = 'https://raw.githubusercontent.com/UoA-eResearch/CoastSat/d61c2052a4a0b9ed9763c6fb89fa7cabdba034ff/transects_extended.geojson'

shore_df = gpd.read_file(file_path)
print(shore_df.head())
#Need to filter NZ locations only

#Pick origin point (landward) coordinates
#retrieves first point (index 0) of LineString 
#"geometry" is where coordinates are stored
land_coord= get_point(shore_df.geometry, 0)

coastsat_coords = {
    'lon':land_coord.x ,
    'lat': land_coord.y
}
#Create DataFrame
coastsat=pd.DataFrame(coastsat_coords)

#To extract beach slope
# bs = gdf.beach_slope

#%% Sea Level rise data
#First read lat lon coordinates, only in VLM file, version 3 of zenodo rep

url_latlon= "https://zenodo.org/records/11398538/files/NZ_VLM_final_May24.csv"
df_latlon= pd.read_csv(url_latlon)
print(df_latlon.head())

df_latlon.iloc[0]

nzsearise_coords = {
    'lon': df_latlon['Lon'],    
    'lat':df_latlon['Lat'] 
}

nzrise = pd.DataFrame(nzsearise_coords)


#%% Ball tree using Haversine metric

# --- Convert lat/lon to radians ---
coastsat_radians = np.deg2rad(coastsat[['lat', 'lon']].values)
nzrise_radians = np.deg2rad(nzrise[['lat', 'lon']].values)

# --- Build BallTree using nzrise coordinates ---
tree = BallTree(nzrise_radians, metric='haversine')

# --- Query nearest neighbor for each coastsat point ---
distances, indices = tree.query(coastsat_radians, k=1)  # k=1 for nearest

# --- Convert radians to meters (Earth radius = ~6371 km) ---
distances_km = distances * 6371  # Convert to km
nearest_indices = indices[:, 0]

df_dis= pd.DataFrame(data=distances_km)

#%%
# --- Get nearest nzrise rows ---
nearest_nzrise = nzrise.iloc[nearest_indices].reset_index(drop=True)


nearest_nzrise.columns = [f'nzrise_{col}' for col in nearest_nzrise.columns]

# --- Combine everything ---
coastsat = coastsat.reset_index(drop=True)
coastsat['distance_km'] = distances_km
combined = pd.concat([coastsat, nearest_nzrise], axis=1)

# # --- Save or inspect ---
# combined.to_csv('coastsat_nearest_nzrise_balltree.csv', index=False)
# print(combined.head())


























