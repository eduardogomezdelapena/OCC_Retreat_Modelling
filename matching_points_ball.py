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
# all_tgroups_2005 = []
# #  CRS consistent ( NZTM 2193)
# target_crs = 2193

# #Loop through all NZ site id's
# for site_id in tqdm(transects.site_id.unique()):
#     site = transects[transects.site_id == site_id]
#     site.set_index("id", inplace=True)

#     #Read tidally corrected ts for each transect
#     intersects = pd.read_csv(f"https://uoa-eresearch.github.io/CoastSat/data/{site_id}/transect_time_series_tidally_corrected.csv")
#     mean_intersect = intersects[intersects.dates.between("2005-01-01", "2006-01-01")].drop(columns=["dates", "satname"]).mean()

#     site.to_crs(target_crs, inplace=True)

#     #All points in a single group transect
#     for transect_id, transect in site.iterrows():
#         all_tgroups_2005.append({
#                         "site_id": site_id,
#                         "transect_id": transect_id,
#                         "geometry": line_interpolate_point(transect.geometry, mean_intersect[transect_id])
#                     })

# # Create GeoDataFrame
# shoreline_2005_gdf = gpd.GeoDataFrame(all_tgroups_2005, crs=target_crs)

# # Preview
# shoreline_2005_gdf.head()

# shoreline_2005_gdf.describe()

shoreline_2005_gdf.to_crs(4326).to_file('points_ref_shoreline_2005.geojson')

shoreline_2005_gdf= gpd.read_file('points_ref_shoreline_2005.geojson')
shoreline_2005_gdf = shoreline_2005_gdf.to_crs(epsg=2193)
# shoreline_2005_gdf.crs
# transects.crs   
# ref_trial.to_crs(epsg=2193).crs

#%% Plot all transects and ref points in map

# Convert data to Web Mercator (EPSG:3857) for plotting with basemap
shoreline_web_mercator = shoreline_2005_gdf.to_crs(epsg=3857)
transects_web_mercator = transects.to_crs(epsg=3857)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot shoreline and transects in the correct projection
shoreline_web_mercator.plot(ax=ax, color='red', markersize=5, label="2005 Shoreline Points")
transects_web_mercator.plot(ax=ax, color='blue', linewidth=0.5, alpha=0.3, label="Transects")

# Add the basemap
ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldImagery)

# Customize the plot
ax.set_title("2005 Shoreline Points Across All NZ Sites")
ax.legend()
ax.set_axis_off()

# Show the map
plt.show()
#%% From points to linestrings

missing = shoreline_2005_gdf[shoreline_2005_gdf.geometry.isnull()]
print(f"Missing geometries: {len(missing)}")

shoreline_2005_gdf = shoreline_2005_gdf.dropna(subset=['geometry'])

# Step 1: Extract group and order fields
shoreline_2005_gdf["group_id"] = shoreline_2005_gdf["transect_id"].str.split("-").str[0]
shoreline_2005_gdf["order_id"] = shoreline_2005_gdf["transect_id"].str.split("-").str[1].astype(int)

# Step 2: Create LineStrings per group
lines = []

for group_id, group in shoreline_2005_gdf.groupby("group_id"):
    sorted_group = group.sort_values(by="order_id")
    coords = sorted_group.geometry.tolist()
    
    # Ensure we have at least 2 points to make a line
    if len(coords) >= 2:
        line = LineString(coords)
        lines.append({"geometry": line, "group_id": group_id})

# Step 3: Create GeoDataFrame of LineStrings
lines_gdf = gpd.GeoDataFrame(lines, crs=shoreline_2005_gdf.crs)

# Step 4: Reproject to WGS84 (EPSG:4326) for web tools like Leaflet / Google Earth
lines_gdf = lines_gdf.to_crs(epsg=4326)

# Step 5: Export to GeoJSON
lines_gdf.to_file("lines_ref_shoreline_2005.geojson", driver="GeoJSON")

lines_gdf.crs
shoreline_2005_gdf.crs

#%%

# #Transform into coastsat df for next steps

#Reproject to Lat Lon (NZsearise data is in lat lon)
# Convert to WGS84 (lat/lon) .to_crs('EPSG:4326')

coastsat_coords = {
    'lon':shoreline_2005_gdf.to_crs('EPSG:4326').geometry.x ,
    'lat': shoreline_2005_gdf.to_crs('EPSG:4326').geometry.y
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

combined['site_ID_nzrise'] = df_latlon['Site ID'].iloc[nearest_indices].reset_index(drop=True)

combined['beach_slope'] = transects.beach_slope.reset_index(drop=True)

#Adding coastsat id tag
combined['coastsat_id'] = transects.id.reset_index(drop=True)

#Adding historic trend
combined['trend'] = transects.trend.reset_index(drop=True)

#%%
#Download actual SLR and VLM csv
# url_slr="https://zenodo.org/records/14722058/files/NZ_Searise_noVLM-2005.csv"

# df_nzrise_slr= pd.read_csv(url_slr)

df_nzrise_slr= pd.read_csv('NZ_Searise_noVLM-2005.csv')
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
#Bruun rule applied
# Shoreface slope multiplied by 0.5 to obtain 
# Bruun beach slope to half of the foreshore beach slope 
# # a bit ad hoc, applied in Vitousek et al. (2023)
#  but is somewhat consistent with Lidar profiles in California

retreat_df = slr_df.div(merged_df['beach_slope'] * 0.5, axis=0)

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
unique_years =  [2005]
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
            site
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
                new_distance = ref_distance_along_line + retreat_distance

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
