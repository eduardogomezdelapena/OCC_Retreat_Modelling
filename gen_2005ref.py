#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create 2005 reference shoreline in points and polyline
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

# Preview
shoreline_2005_gdf.head()

shoreline_2005_gdf.describe()

shoreline_2005_gdf = shoreline_2005_gdf.dropna(subset=['geometry'])
#%%
# Ensure label continuity check within transects. 
# Sometimes labels 0000-0001 are not consecutive.
def check_consecutive_labels(shoreline_2005_gdf):
    """Sometimes labels 0000-0001 are not consecutive, flag if not csctive. """
    # Ensure CRS is projected in meters
    assert shoreline_2005_gdf.crs.to_epsg() == 2193, "Need projected CRS in meters"

    results = []

    for site_id, group in shoreline_2005_gdf.groupby("site_id"):
        # Look up transect 0 and transect 1
        t0 = group[group.transect_id.str.endswith("0000")]
        t1 = group[group.transect_id.str.endswith("0001")]

        if not t0.empty and not t1.empty:
            p0 = t0.iloc[0].geometry
            p1 = t1.iloc[0].geometry

            dist_km = p0.distance(p1) / 1000
            results.append({
                "site_id": site_id,
                "transect0_id": t0.iloc[0].transect_id,
                "transect1_id": t1.iloc[0].transect_id,
                "dist_km": dist_km,
                "flag": dist_km > 2
            })

    site_pair_distances = pd.DataFrame(results)
    return site_pair_distances


def reindex_transects(group: gpd.GeoDataFrame, flagged: bool) -> gpd.GeoDataFrame:
    """Reindex transect IDs within a site if flagged."""
    if not flagged:
        return group

    # Sort transects by their numeric suffix
    group = group.copy()
    group["suffix"] = group.transect_id.str.extract(r"-(\d+)$").astype(int)

    # Rotate suffixes: move 0 to the end, shift others down
    max_suffix = group["suffix"].max()
    group.loc[group["suffix"] == 0, "suffix"] = max_suffix + 1
    group["suffix"] = group["suffix"] - 1  # shift everything up by one

    # Rebuild transect_id string with padded suffix
    group["transect_id"] = group["site_id"] + "-" + group["suffix"].astype(str).str.zfill(4)

    return group.drop(columns="suffix")

#% Plot all transects and ref points in map
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

#% From points to linestrings
def points_to_lines(shoreline_2005_gdf):

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
#%%

site_pair_distances = check_consecutive_labels(shoreline_2005_gdf)
# Show flagged sites
site_pair_distances[site_pair_distances.flag]

# Join flags back
shoreline_flagged = shoreline_2005_gdf.merge(
    site_pair_distances[["site_id", "flag"]],
    on="site_id",
    how="left"
)

# Apply reindexing per site
shoreline_reindexed = (
    shoreline_flagged.groupby("site_id", group_keys=False)
    .apply(lambda g: reindex_transects(g, g["flag"].iloc[0]))
    .drop(columns="flag")   # drop flag here
)

######################Export files#######################
#Export points
shoreline_reindexed.to_crs(4326).to_file('points_ref_shoreline_2005.geojson')

#Export polylines
lines_gdf = points_to_lines(shoreline_reindexed)
lines_gdf.to_file("lines_ref_shoreline_2005.geojson")
# %%
