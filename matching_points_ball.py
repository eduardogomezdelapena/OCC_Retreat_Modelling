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
from sklearn.neighbors import BallTree
import math
from shapely.geometry import Point, LineString, Polygon
from shapely import line_interpolate_point, get_point
import contextily as ctx

#%% #Coastsat data


transects = gpd.read_file("https://uoa-eresearch.github.io/CoastSat/transects_extended.geojson")
#Trim it to just NZ, CoastSat is for the entire Pacific
# Filter rows where 'id' contains 'nzd'
transects = transects[transects.site_id.str.startswith("nzd")]
transects



#%%
#Recalculate reference shoreline point in year 2005


#% MWE for nzd0014
site_id = "nzd0014"
site = transects[transects.site_id == site_id]
site.set_index("id", inplace=True)

intersects = pd.read_csv(f"https://uoa-eresearch.github.io/CoastSat/data/{site_id}/transect_time_series_tidally_corrected.csv")
intersects
#%%
mean_intersect = intersects[intersects.dates.between("2005-01-01", "2006-01-01")].drop(columns=["dates", "satname"]).mean()
# distance = mean_intersect - ((site.retreat_50 / 25) * (2100 - 2005))
# distance

site.to_crs(2193, inplace=True)

points_2005 = []
# points_2100 = []
for transect_id, transect in site.iterrows():
    print(transect_id)
    print("transect:")
    print(transect)
    points_2005.append(line_interpolate_point(transect.geometry, mean_intersect[transect_id]))
    # points_2100.append(line_interpolate_point(transect.geometry, distance[transect_id]))
points_2005#, points_2100




#%%
ax = site.plot(figsize=(10,10))
gpd.GeoSeries(points_2005, crs=2193).plot(ax=ax, color="blue")
ctx.add_basemap(ax, crs=site.crs.to_string(), source=ctx.providers.Esri.WorldImagery)


#Transform into coastsat df for next steps
lol= gpd.GeoSeries(points_2005, crs=2193)
lol.geometry


#Pick origin point (landward) coordinates
#retrieves first point (index 0) of LineString 
#"geometry" is where coordinates are stored
land_coord= get_point(transects.geometry, 0)

coastsat_coords = {
    'lon':land_coord.x ,
    'lat': land_coord.y
}
#Create DataFrame
coastsat=pd.DataFrame(coastsat_coords)


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


#%% 
#SLR from nearest NZRise point
#First retreive SiteID

lol=df_latlon['Site ID'].iloc[nearest_indices]

combined['site_ID_nzrise'] = df_latlon['Site ID'].iloc[nearest_indices].reset_index(drop=True)

combined['beach_slope'] = shore_df.beach_slope.reset_index(drop=True)

#Adding coastsat id tag
combined['coastsat_id'] = shore_df.id.reset_index(drop=True)

#Adding historic trend
combined['trend'] = shore_df.trend.reset_index(drop=True)

#%%
#Download actual SLR and VLM csv
url_slr="https://zenodo.org/records/14722058/files/NZ_Searise_noVLM-2005.csv"

df_nzrise_slr= pd.read_csv(url_slr)
print(df_nzrise_slr.head())
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
#Bruun rule
retreat_df = slr_df.div(merged_df['beach_slope'], axis=0)

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
unique_years =  [2005, 2020, 2030, 2050, 2080, 2100]
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

            #Transform to geopandas df
            gdf = gpd.GeoDataFrame(subset, geometry=geometry)
            # Set coordinate reference system (CRS)
            gdf.set_crs(epsg=2193, inplace=True)  # WGS84
            # Export
            gdf.to_file(url_sv_gj+filename, driver="GeoJSON")
            print(filename+' saved ')





















