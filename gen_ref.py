#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean CoastSat transects & create reference shoreline (custom year), export points and polyline.
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
import time
#%%
# Parameters
slrise_ref = 2005
custom_ref_year = 2025
CRS_NZTM = 2193  # NZ Transverse Mercator
CRS_WGS84 = 4326 # Lat/Lon

#%%
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

def points_to_lines(shoreline_ref):
    """ Transform points to polylines, groupped by transect id"""
    missing = shoreline_ref[shoreline_ref.geometry.isnull()]
    print(f"Missing geometries: {len(missing)}")
    # Extract group and order fields
    shoreline_ref["group_id"] = shoreline_ref["transect_id"].str.split("-").str[0]
    shoreline_ref["order_id"] = shoreline_ref["transect_id"].str.split("-").str[1].astype(int)
    #Create LineStrings per group
    lines = []
    for group_id, group in shoreline_ref.groupby("group_id"):
        sorted_group = group.sort_values(by="order_id")
        coords = sorted_group.geometry.tolist()
        # Ensure we have at least 2 points to make a line
        if len(coords) >= 2:
            line = LineString(coords)
            lines.append({"geometry": line, "group_id": group_id})
    #Create GeoDataFrame of LineStrings
    lines_gdf = gpd.GeoDataFrame(lines, crs=shoreline_ref.crs)
    #Reproject to WGS84 (EPSG:4326) for web tools like Leaflet / Google Earth
    lines_gdf = lines_gdf.to_crs(epsg=4326)
    return lines_gdf

def sanity_plot(transects, shoreline_ref, custom_ref_year):
    """ Plots extracted shoreline and reference transects"""
    # Convert data to Web Mercator (EPSG:3857) for plotting with basemap
    shoreline_web_mercator = shoreline_ref.to_crs(epsg=3857)
    transects_web_mercator = transects.to_crs(epsg=3857)
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    # Plot shoreline and transects in the correct projection
    shoreline_web_mercator.plot(ax=ax, color='red', markersize=5, label="2005 Shoreline Points")
    transects_web_mercator.plot(ax=ax, color='blue', linewidth=0.5, alpha=0.3, label="Transects")
    # Add the basemap
    ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldImagery)
    # Customize the plot
    ax.set_title(f" {custom_ref_year} Shoreline Points Across All NZ Sites")
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

#Hardcoded: drop nested transects near Rotokino mouth
transects = transects[transects.site_id != 'nzd0418']
transects = transects[transects.site_id != 'nzd0419']

#Hardcoded: drop Farewell spit transects
transects = transects[transects.site_id != 'nzd0313']
transects = transects[transects.site_id != 'nzd0314']

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

###################################################################################################
#%%  Generate shoreline points, reference year: custom

all_trgoups_ref = []
#Loop through all NZ site id's
for site_id in tqdm(transects_reindexed.site_id.unique()):
    site = transects_reindexed[transects_reindexed.site_id == site_id]
    site.set_index("id", inplace=True)
    site = site.to_crs(CRS_NZTM) 
    #Read tidally corrected ts for each transect
    intersects = pd.read_csv(f"https://uoa-eresearch.github.io/CoastSat/data/{site_id}/transect_time_series_tidally_corrected.csv")
    mean_intersect = intersects[intersects.dates.between(str(custom_ref_year-1), str(custom_ref_year))].drop(columns=["dates", "satname"]).mean(
        skipna=True
    )
    #All points in a single group transect
    for transect_id, transect in site.iterrows():
        all_trgoups_ref.append({
                        "site_id": site_id,
                        "transect_id": transect_id,
                        "geometry": line_interpolate_point(transect.geometry, mean_intersect[transect_id])
                    })
# Create GeoDataFrame
shoreline_ref = gpd.GeoDataFrame(all_trgoups_ref, crs=CRS_NZTM)
##################################################################################################
#%% Export points
shoreline_ref = shoreline_ref.dropna(subset=['geometry'])
shoreline_ref.to_crs(CRS_WGS84).to_file(f'points_ref_shoreline_{custom_ref_year}.geojson')


#Intersect with rivers






#Export polylines
lines_gdf = points_to_lines(shoreline_ref)
lines_gdf.to_file(f"lines_ref_shoreline_{custom_ref_year}.geojson")
#Export clean transects 
transects_reindexed.to_file("transects_reindexed.geojson")

# %%

def load_slrdata_nzrise(filepath:str):
    """Load SLR data from NZRise project. Returns pd.DataFrame"""
    slr_data = (
        pd.read_csv(filepath)
        .rename(columns={"site": "nzrise_site_id"})
    )
    return slr_data

slr_fp = "NZ_Searise_noVLM-2005"
slr_data  = load_slrdata_nzrise(f"{slr_fp}.csv") #pd.DataFrame



#%% Adjust the slr data

# Record start time
start_time = time.time()

# Precompute available years and bounds
available_years = sorted(slr_data["year"].unique())
lower = max([y for y in available_years if y <= custom_ref_year])
upper = min([y for y in available_years if y >= custom_ref_year])

print(f"Using lower year: {lower}, upper year: {upper}")
print("assuming linear change between lower and upper year bounds")

# Prepare shifted columns
for q in ["17", "50", "83"]:
    shifted_col = f"{q}_shifted"
    if shifted_col not in slr_data.columns:
        slr_data[shifted_col] = np.nan

# Group by unique combinations of site + SSP + scenario + confidence
group_cols = ["nzrise_site_id", "SSP", "scenario", "Confidence"]
groups = slr_data.groupby(group_cols)

# tqdm progress bar
pbar = tqdm(total=len(groups), desc="Processing slr data correction to custom ref year", ncols=100)

# Loop over grouped data
for (site_id, ssp_val, scenario_val, confidence_val), group in groups:

    # Get lower and upper rows once
    lower_row = group.loc[group["year"] == lower]
    upper_row = group.loc[group["year"] == upper]

    # Loop through quartiles
    for q in ["17", "50", "83"]:
        R_lower = lower_row[q].values[0]
        R_upper = upper_row[q].values[0]
        Rdecade = R_upper - R_lower
        Rperyear = Rdecade / (upper - lower)
        correction_factor = R_lower + Rperyear * (custom_ref_year - lower)

        # Compute shifted values and assign directly via index alignment
        slr_data.loc[group.index, f"{q}_shifted"] = group[q] - correction_factor

    pbar.update(1)

# Record end time
end_time = time.time()
pbar.close()
print(f"\n✅ Processing complete: slr data adjusted to ref year: {custom_ref_year}.")

# Compute elapsed time in seconds
elapsed_seconds = end_time - start_time

print(f"\n✅ Total processing time: {elapsed_seconds/60:.2f} minutes")

#Save custom ref year asjusted slr data
slr_data.to_csv(f"{slr_fp}_{custom_ref_year}adjusted.csv", index=False)
print(f"\n💾 Updated dataset saved to: {slr_fp}_{custom_ref_year}adjusted.csv")


#%% Plot for specific site and scenario
site_id = 200
ssp_val = "ssp1"
scenario_val = 2.6
confidence_val = "low_confidence"

# Subset data for that specific combination
slr_subset = slr_data[
    (slr_data["nzrise_site_id"] == site_id)
    & (slr_data["SSP"] == ssp_val)
    & (slr_data["scenario"] == scenario_val)
    & (slr_data["Confidence"] == confidence_val)
].sort_values("year")

# Check if subset exists
if slr_subset.empty:
    print("⚠️ No data found for the selected site and scenario combination.")
else:
    # Create larger figure
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Define quartiles and colors
    quartiles = ["17", "50", "83"]
    colors = {"17": "orange", "50": "red", "83": "green"}

    # Plot original and shifted series for each quartile
    for q in quartiles:
        if q in slr_subset.columns and f"{q}_shifted" in slr_subset.columns:
            ax1.plot(
                slr_subset["year"],
                slr_subset[q],
                color=colors[q],
                label=f"original {q}"
            )
            ax1.plot(
                slr_subset["year"],
                slr_subset[f"{q}_shifted"],
                color=colors[q],
                linestyle="--",
                label=f"shifted {q}"
            )

    # Add vertical line and annotation
    ax1.axvline(x=custom_ref_year, color="blue", linestyle="--", linewidth=1.5)
    ax1.text(
        custom_ref_year + 0.5,
        ax1.get_ylim()[1] * 0.95,
        "custom ref year",
        color="blue",
        fontsize=11,
        rotation=90,
        va="top"
    )

    # Add horizontal line at y = 0
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)

    # Labels and legend
    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Sea level rise (m)", fontsize=12)
    ax1.legend(fontsize=10, ncol=2)

    # Set x-axis ticks every 5 years
    year_min = int(slr_subset["year"].min())
    year_max = int(slr_subset["year"].max())
    ax1.set_xticks(np.arange(year_min, year_max + 1, 5))
    ax1.set_xticklabels(ax1.get_xticks().astype(int), rotation=45, ha="center")

    # 🔹 Dynamic title
    ax1.set_title(
        f"Site {site_id} — SSP: {ssp_val}, Scenario: {scenario_val}, Confidence: {confidence_val}",
        fontsize=13,
        pad=15
    )

    # Improve spacing
    plt.tight_layout()
    plt.show()



