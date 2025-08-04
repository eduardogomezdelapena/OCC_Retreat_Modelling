#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 18:08:50 2025

@author: eegp
"""

#Associate each point in CoastSat CSV with the 
#closest point in NZSeaRise CSV, using a nearest 
#neighbor search based on geographic distance

#Load csv files
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import LineString, get_point

#%% #Coastsat data
file_path = 'https://raw.githubusercontent.com/UoA-eResearch/CoastSat/d61c2052a4a0b9ed9763c6fb89fa7cabdba034ff/transects_extended.geojson'

shore_df = gpd.read_file(file_path)
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
df_shore_coords=pd.DataFrame(coastsat_coords)

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

df_nzsearise_coords = pd.DataFrame(nzsearise_coords)

#%% Pick nearest NZSeaRise point to CoastSat
# (harversine function)

def to_radians(df):
    return np.radians(df[['lat', 'lon']].values)

coords1_rad = to_radians(df_shore_coords)
coords2_rad = to_radians(df_nzsearise_coords)


#%%

from scipy.spatial import cKDTree

# Earth radius in kilometers
EARTH_RADIUS = 6371.0

# Build KDTree on csv1 (in radians)
tree = cKDTree(coords1_rad)

# Query closest point in csv1 for each point in csv2
distances, indices = tree.query(coords2_rad, k=1)

# Convert distance from radians to kilometers
distances_km = distances * EARTH_RADIUS

#%%
csv1= df_shore_coords
csv2= df_nzsearise_coords
# Get nearest csv1 rows
nearest_csv1 = csv1.iloc[indices].reset_index(drop=True)

# Combine with csv2
result = pd.concat([csv2.reset_index(drop=True),
                    nearest_csv1.add_prefix('nearest_'),
                    pd.Series(distances_km, name='distance_km')],
                   axis=1)




#Each row in result now has:

#Original data from csv2

#The closest point from csv1

#The distance (in km) between the points












