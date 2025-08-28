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
#%% #Coastsat data

transects = gpd.read_file("https://uoa-eresearch.github.io/CoastSat/transects_extended.geojson")
#Trim it to just NZ, CoastSat is for the entire Pacific
# Filter where 'id' contains 'nzd'
transects = transects[transects.site_id.str.startswith("nzd")]
transects

#%%  Shoreline position for ref year 2005
all_tgroups_2005 = []
#  CRS consistent ( NZTM 2193)
target_crs = 2193

#Loop through all NZ site id's
for site_id in tqdm(transects.site_id.unique()):
    site = transects[transects.site_id == site_id]
    site.set_index("id", inplace=True)

    #Read tidally corrected ts for each transect
    intersects = pd.read_csv(f"https://uoa-eresearch.github.io/CoastSat/data/{site_id}/transect_time_series_tidally_corrected.csv")
    mean_intersect = intersects[intersects.dates.between("2005-01-01", "2006-01-01")].drop(columns=["dates", "satname"]).mean()

    site.to_crs(target_crs, inplace=True)

    #All points in a single group transect
    for transect_id, transect in site.iterrows():
        all_tgroups_2005.append({
                        "site_id": site_id,
                        "transect_id": transect_id,
                        "geometry": line_interpolate_point(transect.geometry, mean_intersect[transect_id])
                    })

# Create GeoDataFrame
shoreline_2005_gdf = gpd.GeoDataFrame(all_tgroups_2005, crs=target_crs)

# Preview
shoreline_2005_gdf.head()

shoreline_2005_gdf.describe()

shoreline_2005_gdf.to_crs(4236).to_file('2005_ref_points.geojson')

#%%Plot all transect groups in map
fig, ax = plt.subplots(figsize=(12, 12))
shoreline_2005_gdf.plot(ax=ax, color='red', markersize=5, label="2005 Shoreline Points")
transects.to_crs(target_crs).plot(ax=ax, color='blue', linewidth=0.5, alpha=0.3, label="Transects")

ctx.add_basemap(ax, crs=shoreline_2005_gdf.crs, source=ctx.providers.Esri.WorldImagery, zoom='auto')

ax.set_title("2005 Shoreline Points Across All NZ Sites")
ax.legend()
ax.set_axis_off()
plt.show()
#%% Save as linestrings per id group
# Extract group_id prefix from 'coastsat_id' (split by '-')

gdf = shoreline_2005_gdf
gdf.crs
shoreline_2005_gdf.head()

# Extract base route key from route_id

gdf['group_id'] = shoreline_2005_gdf['transect_id'].str.split('-').str[0]

# (Optional) sort if needed, e.g., by route_id
gdf = gdf.sort_values(['group_id', 'transect_id'])

geo_df = gdf.groupby('group_id')['geometry'].apply(lambda x: LineString(x.tolist()))

geo_df = gpd.GeoDataFrame(geo_df2, geometry='geometry')

lines_gdf.set_crs(epsg=2193, inplace=True)
lines_gdf.to_crs(4236).to_file("2005_ref_line.geojson")

#%%
line = LineString(all_tgroups_2005)
gpd.GeoSeries(line, crs=2193).to_crs(4236).to_file("2005_ref_line.geojson")


#%%

#Transform into coastsat df for next steps
lol= gpd.GeoSeries(points_2005, crs=target_crs)
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
            gdf.set_crs(epsg=target_crs, inplace=True)  # WGS84
            # Export
            gdf.to_file(url_sv_gj+filename, driver="GeoJSON")
            print(filename+' saved ')





















