#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean CoastSat transects & create reference shoreline (yr 2005), export points and polyline.
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
def merge_and_renumber(transects: gpd.GeoDataFrame, from_site: str,
                        to_site: str, crs_epsg: int = 2193) -> gpd.GeoDataFrame:
    """
    Merge transects from one site_id into another and renumber IDs in the target site.
    """
    transects.loc[transects.site_id == from_site , "site_id"] = to_site  #change id label
    # Get the index mask for those rows
    mask = transects.site_id == to_site
   # Project to appropriate CRS for accurate distance calculation
    subset = transects[mask].to_crs(epsg=crs_epsg).copy()
    
    # Calculate centroids
    subset['centroid'] = subset.geometry.centroid
    
    # Drop duplicates based on centroid.x (keeping last)
    subset = subset.drop_duplicates(subset=subset['centroid'].x, keep='last')

    # Sort transects by centroid's x coordinate (easting)
    subset = subset.sort_values(by=subset.centroid.x).copy()
    
    # Assign new IDs in geographic order
    subset["id"] = [f"{to_site}-{i:04d}" for i in range(len(subset))]
    
    # Drop centroid helper column before updating
    subset = subset.drop(columns='centroid')
    
    # Update original GeoDataFrame (in original CRS)
    # Note: subset is currently projected, so re-project back
    subset = subset.to_crs(transects.crs)
    transects.update(subset)
    return transects

def drop_close_transects(transects: gpd.GeoDataFrame, 
                         site_id: str, crs_epsg: int = 2193) -> gpd.GeoDataFrame:
    """
    Drop transects within a given site_id where consecutive transect centroids
    are closer than the median distance.
    Reassign IDs after dropping.
    """
    subset = transects[transects.site_id == site_id ].to_crs(epsg=crs_epsg).copy()
    rep_points = subset.centroid   # Get centroids
    # Distances between consecutive centroids
    distances = [
        rep_points.iloc[i].distance(rep_points.iloc[i + 1])
        for i in range(len(rep_points) - 1)
    ]
    median_distance = np.median(distances)
    print(median_distance)
    # Find indices (from the original transects GeoDataFrame) to drop
    drop_indices = [
        subset.index[i + 1]
        for i, dist in enumerate(distances)
        if dist < median_distance * 0.5
    ]
    #Drop from original DataFrame
    transects = transects.drop(index=drop_indices)
    #Reassign IDs in the site after dropping
    updated_subset = transects[transects.site_id == site_id].sort_values(by="geometry").copy()
    updated_subset["id"] = [f"{site_id}-{i:04d}" for i in range(len(updated_subset))]
    transects.update(updated_subset)
    return transects

def check_consecutive_labels(transects: gpd.GeoDataFrame):
    """Sometimes transect labels 0000-0001 are not geograhpically adjacent, flag if not csctive. """
    results = []
    for site_id, group in transects.groupby("site_id"):
        # Look up transect 0 and transect 1
        t0 = group[group.id.str.endswith("0000")]
        t1 = group[group.id.str.endswith("0001")]
        # Get a single point for each transect
        rep_points = group.to_crs(epsg=2193).centroid
        # Distance between consecutive transects
        distances = [
            rep_points.iloc[i].distance(rep_points.iloc[i+1])
            for i in range(len(rep_points)-1)
        ]
        #Median distance between transects
        median_val= np.median(distances)
        #Distance beteen site 00 and 01
        dist_m = distances[0]
        results.append({
            "site_id": site_id,
            "transect0_id": t0.iloc[0].id,
            "transect1_id": t1.iloc[0].id,
            "median" : median_val,
            "dist_m": dist_m,
            "flag": dist_m > (1.5 * median_val)
        })
    return pd.DataFrame(results)

def reindex_transects(group: gpd.GeoDataFrame, flagged: bool) -> gpd.GeoDataFrame:
    """Reindex transect IDs within a site if flagged."""
    if not flagged:
        return group
    # Sort transects by their numeric suffix
    group = group.copy()
    group["suffix"] = group.id.str.extract(r"-(\d+)$").astype(int)
    # Rotate suffixes: move 0 to the end, shift others down
    max_suffix = group["suffix"].max()
    group.loc[group["suffix"] == 0, "suffix"] = max_suffix + 1
    group["suffix"] = group["suffix"] - 1  # shift everything up by one
    # Rebuild id string with padded suffix
    group["id"] = group["site_id"] + "-" + group["suffix"].astype(str).str.zfill(4)
    return group.drop(columns="suffix")

def points_to_lines(shoreline_2005_gdf):
    """ Transform points to polylines, groupped by transect id"""
    missing = shoreline_2005_gdf[shoreline_2005_gdf.geometry.isnull()]
    print(f"Missing geometries: {len(missing)}")
    # Extract group and order fields
    shoreline_2005_gdf["group_id"] = shoreline_2005_gdf["transect_id"].str.split("-").str[0]
    shoreline_2005_gdf["order_id"] = shoreline_2005_gdf["transect_id"].str.split("-").str[1].astype(int)
    #Create LineStrings per group
    lines = []
    for group_id, group in shoreline_2005_gdf.groupby("group_id"):
        sorted_group = group.sort_values(by="order_id")
        coords = sorted_group.geometry.tolist()
        # Ensure we have at least 2 points to make a line
        if len(coords) >= 2:
            line = LineString(coords)
            lines.append({"geometry": line, "group_id": group_id})
    #Create GeoDataFrame of LineStrings
    lines_gdf = gpd.GeoDataFrame(lines, crs=shoreline_2005_gdf.crs)
    #Reproject to WGS84 (EPSG:4326) for web tools like Leaflet / Google Earth
    lines_gdf = lines_gdf.to_crs(epsg=4326)
    return lines_gdf

def sanity_plot(transects, shoreline_2005_gdf):
    """ Plots extracted shoreline and reference transects"""
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
#####################################################################################################
#%%  Import Coastsat transects
transects = gpd.read_file("https://uoa-eresearch.github.io/CoastSat/transects_extended.geojson")
#Trim it to just NZ, CoastSat is for the entire Pacific
transects = transects[transects.site_id.str.startswith("nzd")]
#%% CLEANING TRANSECTS
# Find which ids are duplicated
dupe_ids = transects['id'][transects['id'].duplicated()].unique()
print("Duplicate transect IDs:")
print(dupe_ids)


#%%

from_site="nzd0418"
to_site="nzd0419"
crs_epsg: int = 2193

transects.loc[transects.site_id == from_site , "site_id"] = to_site  #change id label
# Get the index mask for those rows
mask = transects.site_id == to_site
# Project to appropriate CRS for accurate distance calculation
subset = transects[mask].to_crs(epsg=crs_epsg).copy()

# Calculate centroids
subset['centroid'] = subset.geometry.centroid
subset['centroid_x'] = subset.geometry.centroid.x

# Drop duplicates based on centroid.x (keeping last)
subset = subset.drop_duplicates(subset=['centroid_x'], keep='last')

# Sort transects by centroid's x coordinate (easting)
subset = subset.sort_values(by='centroid_x').copy()

# Assign new IDs in geographic order
subset["id"] = [f"{to_site}-{i:04d}" for i in range(len(subset))]

# Drop centroid helper column before updating
subset = subset.drop(columns=['centroid','centroid_x'])

# Update original GeoDataFrame (in original CRS)
# Note: subset is currently projected, so re-project back
subset = subset.to_crs(transects.crs)
transects.update(subset)
   
transects.to_file("trial_transects_reindexed.geojson")


#%%



# Merge overlapping sites and renumber
# transects = merge_and_renumber(transects, from_site="nzd0418", to_site="nzd0419")   

#Export clean transects 


#%%
#Now drop 
print("Before dropping:")
print(transects[transects.site_id == 'nzd0419'][['site_id', 'id']].sort_values('id'))

transects = drop_close_transects(transects, "nzd0419")
# After dropping
print("\nAfter dropping:")
print(transects[transects.site_id == 'nzd0419'][['site_id', 'id']].sort_values('id'))

#Consecutive label check (ocasional bug between 000-001 ids)
site_pair_distance = check_consecutive_labels(transects)
#  Show flagged sites
site_pair_distance[site_pair_distance.flag]
# Join flags back
transects_flagged = transects.merge(
    site_pair_distance[["site_id", "flag"]],
    on="site_id",
    how="left"
)
# Apply reindexing per site
transects_reindexed = (
    transects_flagged.groupby("site_id", group_keys=False)
    .apply(lambda g: reindex_transects(g, g["flag"].iloc[0]))
    .drop(columns="flag")   # drop flag here
)

#Export clean transects 
transects_reindexed.to_file("trial_transects_reindexed.geojson")

###################################################################################################
#%%  Generate shoreline points, reference year: 20005
all_tgroups_2005 = []
#  CRS consistent ( NZTM 2193)
target_crs = 2193
#Loop through all NZ site id's
for site_id in tqdm(transects_reindexed.site_id.unique()):
    site = transects_reindexed[transects_reindexed.site_id == site_id]
    site.set_index("id", inplace=True)
    site = site.to_crs(target_crs) 
    #Read tidally corrected ts for each transect
    intersects = pd.read_csv(f"https://uoa-eresearch.github.io/CoastSat/data/{site_id}/transect_time_series_tidally_corrected.csv")
    mean_intersect = intersects[intersects.dates.between("2005-01-01", "2006-01-01")].drop(columns=["dates", "satname"]).mean(
        skipna=True
    )
    #All points in a single group transect
    for transect_id, transect in site.iterrows():
        all_tgroups_2005.append({
                        "site_id": site_id,
                        "transect_id": transect_id,
                        "geometry": line_interpolate_point(transect.geometry, mean_intersect[transect_id])
                    })
# Create GeoDataFrame
shoreline_2005_gdf = gpd.GeoDataFrame(all_tgroups_2005, crs=target_crs)
##################################################################################################
#%% Export points
shoreline_2005_gdf = shoreline_2005_gdf.dropna(subset=['geometry'])
shoreline_2005_gdf.to_crs(4326).to_file('points_ref_shoreline_2005.geojson')
#Export polylines
lines_gdf = points_to_lines(shoreline_2005_gdf)
lines_gdf.to_file("lines_ref_shoreline_2005.geojson")
#Export clean transects 
transects_reindexed.to_file("transects_reindexed.geojson")

# %%
