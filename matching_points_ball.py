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
# print(shore_df.head())

#Trim it to just NZ, CoastSat is for the entire Pacific
# Filter rows where 'id' contains 'nzd'
shore_df = shore_df[shore_df['id'].str.contains('nzd', na=False)]
print(shore_df.head())


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
#https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b/

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
coastsat_re = coastsat.reset_index(drop=True)

combined = pd.concat([nearest_nzrise, coastsat_re], axis=1)

combined['distance_km'] = distances_km

# # --- Save or inspect ---
# combined.to_csv('coastsat_nearest_nzrise_balltree.csv', index=False)
# print(combined.head())


threshold = 2
# More than 2 km in difference
count = (combined['distance_km'] > threshold).sum()


#%% Coarse calculation Bruun Rule

#Beach slope
shore_df.beach_slope

#SLR from nearest NZRise point
#First retreive SiteID

lol=df_latlon['Site ID'].iloc[nearest_indices]

combined['site_ID_nzrise'] = df_latlon['Site ID'].iloc[nearest_indices].reset_index(drop=True)

combined['beach_slope'] = shore_df.beach_slope.reset_index(drop=True)

#%%
#Download actual SLR and VLM csv
url_slr="https://zenodo.org/records/14722058/files/NZ_Searise_noVLM-2005.csv"

df_nzrise_slr= pd.read_csv(url_slr)
print(df_nzrise_slr.head())

#Filter year = 2030, and ssp1, then match site with siteId_NZrise

unique_ssp_values = df_nzrise_slr['SSP'].unique()

unique_scenarios = df_nzrise_slr['scenario'].unique()

unique_years = df_nzrise_slr['year'].unique()

lol = df_nzrise_slr[df_nzrise_slr['year']== 2030]

lol = lol[lol['SSP'].str.contains('ssp1', na=False)]

#50th percentile (mean value) for projections

lol= lol.drop(['17','83'],axis=1)

# Step 1: Remove duplicates in loldataframe based on 'site'
lol_unique = lol.drop_duplicates(subset='site')

# Step 2: Map the '50' column to combineddataframe based on matching site IDs
combined['SLR_ssp1_2030_50p'] = combined['site_ID_nzrise'].map(
    lol_unique.set_index('site')['50']
)

bruun_retreat= (1/ combined.beach_slope) * combined.SLR_ssp1_2030_50p


combined['bruun_retreat1'] = bruun_retreat
combined['bruun_retreat2'] = bruun_retreat

#%% Convert to geojson
import geopandas as gpd
from shapely.geometry import Point

# Convert to geometry
geometry = [Point(xy) for xy in zip(combined['lon'], combined['lat'])]

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(combined, geometry=geometry)

# Set coordinate reference system (CRS)
gdf.set_crs(epsg=4326, inplace=True)  # WGS84

url_sv_gj="/home/eegp/Documents/GitHub/OCC_Retreat_Modelling/"

gdf.to_file(url_sv_gj+"shoreline_retreat.geojson", driver="GeoJSON")
























