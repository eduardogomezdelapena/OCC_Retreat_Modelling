#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep  4 2025

@author: eduardo.gomez
"""
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

from sklearn.neighbors import BallTree
import math
from shapely.geometry import Point, LineString, Polygon
from shapely import line_interpolate_point, get_point
from shapely.ops import nearest_points

import contextily as ctx
from tqdm import tqdm  # progress bar
from shapely.errors import GEOSException
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter

import time

from smooth_algorithms.smoothn import smoothn

#%%
slrise_ref = 2005
custom_ref_year = 2025
retreat_scen_marker ="oslr"
# retreat_scen_marker = ["oslr","htrend"]

# Constants
EARTH_RADIUS_KM = 6371.0
CRS_NZTM = 2193  # NZ Transverse Mercator
CRS_WGS84 = 4326 # Lat/Lon

#Fix randomness
np.random.seed(42)
#%%
def load_transects(filepath: str):
    """Load and filter transects (NZ only)."""
    transects = gpd.read_file(filepath)
    transects = transects.drop_duplicates(subset="id") #drop duplicated transects
    return transects[transects["site_id"].str.startswith("nzd")]

def load_points(filepath: str):
    """Load ref points (shoreline) and reproject, drop missing geometries.
    Points externally generated in gen_ref.py"""
    points = gpd.read_file(filepath)
    return points.dropna(subset=["geometry"])

def rename_columns(transects: gpd.GeoDataFrame,
                   points: gpd.GeoDataFrame):
    """Rename columns for consistency."""
    transects = transects.rename(columns={
        "site_id": "coastsat_site_id",
        "id": "coastsat_transect_id",
        "geometry": "geom_transect_coastsat"
    })
    points = points.rename(columns={
        "site_id": "coastsat_site_id",
        "transect_id": "coastsat_transect_id",        
        "geometry": "geom_points_ref"
    })
    return transects, points

def load_and_merge_coastsat_data(transects_fp: str,
                                 points_fp: str,
                                 crs_str: int,
                                 col_merge_label: str = "coastsat_transect_id"):
    """Wrapper: load, clean, and merge coastsat data."""
    transects = load_transects(transects_fp)
    points = load_points(points_fp)
    transects, points = rename_columns(transects, points)

    # Merge on a column
    merged = pd.merge(transects,
                      points[[
                          c for c in points.columns if c not in transects.columns or c == col_merge_label]],
                      on= col_merge_label,  how="inner")
    merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{crs_str}", geometry= 'geom_points_ref')

        # Drop transects belonging to sites where all slopes are NaN
    empty_sites = merged.groupby('coastsat_site_id')['beach_slope'].apply(lambda x: x.isna().all())
    merged = merged[~merged['coastsat_site_id'].isin(empty_sites[empty_sites].index)]
    print(f"Sites where all slopes are NaN{empty_sites[empty_sites].index.tolist()}, deleted")

    empty_trend_sites = merged.groupby('coastsat_site_id')['trend'].apply(lambda x: x.isna().all())
    merged = merged[~merged['coastsat_site_id'].isin(empty_trend_sites[empty_trend_sites].index)]
    print(f"Sites where all trends are NaN{empty_trend_sites[empty_trend_sites].index.tolist()}, deleted")
      
    print(f"Merged {len(merged)} transects/points")

    return merged.reset_index(drop=True)
# %
def load_metadata_nzrise(filepath:str, crs_str: int):
    """Load metadata for NZRise. Return GeoDataFrame"""
    df = (
        pd.read_csv(filepath)
        .rename(columns={"Site ID": "nzrise_site_id"})
    )
    # Drop entirely empty columns
    df = df.dropna(axis=1, how="all")
    # Create geometry column from latitude and longitude
    geometry = [Point(xy) for xy in zip(df['Lon'], df['Lat'])]
    meta_data = gpd.GeoDataFrame(df, geometry=geometry)
    meta_data.drop(columns=["Lon","Lat"], inplace=True)
    meta_data.rename_geometry("geom_nzrise",inplace=True)
    meta_data.set_crs(epsg=crs_str,inplace=True)
    return meta_data

def load_slrdata_nzrise(filepath:str):
    """Load SLR data from NZRise project. Returns pd.DataFrame"""
    slr_data = (
        pd.read_csv(filepath)
        .rename(columns={"site": "nzrise_site_id"})
    )
    return slr_data

#%
def nearest_points(meta_data, coastsat_merged):
    """ Find nearest NZRise points to CoastSat data"""

    #  Nearest neighbor search with BallTree
    # Extract lon/lat from geometries
    coastsat_coords   = np.column_stack((coastsat_merged.geometry.y, coastsat_merged.geometry.x))    
    nzrise_coords = np.column_stack((meta_data.geometry.y, meta_data.geometry.x))

    # Convert to radians
    coastsat_rad = np.deg2rad(coastsat_coords)
    nzrise_rad   = np.deg2rad(nzrise_coords)

    # Build BallTree
    tree = BallTree(nzrise_rad, metric="haversine")

    # Query nearest neighbor
    distances, indices = tree.query(coastsat_rad, k=1)
    distances_km = distances.flatten() * EARTH_RADIUS_KM
    nearest_indices = indices.flatten()

    # Flag cases farther than 2 km 
    threshold = 2; count = (distances_km > threshold).sum()
    print(f"Points farther than {threshold} km: {count}")

    return distances, nearest_indices

def calc_retreat(all_merged, retreat_scen_marker,  c_adjust = 0.5, custom_ref_year= custom_ref_year ):
    """ Shoreline retreat calculation. Data processing of slope and trend data. Bruun rule application.
    Two possible scenarios only sea level rise (oslr), or slr + historic trend (htrend),determined
    by string retreat_scen_marker"""

    def fillna_mildslop_smooth(slope):
        """ Slope data processing and cleaning.
        1: Fill NaNs with group mean. 
        2: Fill in very mildly sloping transects with mean.
        3: Smooth with signal processing filter.
        4: Apply perturbations, to avoid chunks of coast with same mean tanb value  """

        # Fill NaNs with the group-wise mean
        mean_val = np.nanmean(slope)
        if np.isnan(mean_val):
            site_id = slope.name  # this is the current group key from groupby
            print(f"⚠️ Warning: all NaN slope values for site_id = {site_id}")
    
        slope = slope.fillna(mean_val)

        # Replace values where 1 / slope > 60 
        inv_condition = (slope != 0) & ((1 / slope) > 60)
        if inv_condition.any():
            slope[inv_condition] = mean_val

        # Apply low-pass filter to smooth lonshore variability, if enough points
        if len(slope) > 6:
            b, a = butter(2, 0.01, btype='low', analog=False) #filter to smooth longshore variability
            slope = pd.Series(filtfilt(b, a, slope), index=slope.index)
        
        #Apply perturbations to beach slope
        slope = slope * (1 + 0.05 * np.random.randn(*slope.shape))

        return slope
    
    def smoothn_by_variability(x):
        """ Satellite trend smoothing. Garcia smoother, dynamic smoothing factor s
        varies with the standard deviation of the trend spatial series """

        site_id = x.name
        std_x = x.std()

        # Skip smoothing if variability is very low
        if std_x < 0.01:
            print(f"ℹ️ Skipping smoothing for site_id = {site_id} (std = {std_x:.4f})")
            return x.to_numpy()
    
        s = 100000 if std_x > 0.2 else 10000 # s a bit adhoc, depends on stdev
    
        return smoothn(x.to_numpy(), isrobust=True, s=s)[0]


    # slr_quantiles = ["17", "50", "83"]
    slr_quantiles = ["17_shifted", "50_shifted", "83_shifted"]
     
    all_merged['beach_slope'] = all_merged.groupby('coastsat_site_id')['beach_slope']\
                                      .transform(fillna_mildslop_smooth)

    all_merged['trend'] = all_merged.groupby('coastsat_site_id')['trend']\
            .transform(smoothn_by_variability)
    
    # Apply Bruun rule
    # c_adjust = 0.5 to adjust the Bruun profile with the shoreface profile
    # a bit ad hoc, matches with some lidar measurements
    denom = c_adjust * all_merged["beach_slope"]

    retreat_df = (
        all_merged[slr_quantiles]
        .div(denom , axis=0)
        .rename(columns=lambda c: f"retreat_{c}")
        .round(2)  # ensures 3 decimals
    )

    if retreat_scen_marker == 'oslr':
        # Append retreat columns
        merged_retreat_df = pd.concat([all_merged, retreat_df], axis=1)

    elif retreat_scen_marker == 'htrend': 
        #(Optional) Historic rate adjustment. Trend is in (m/year)
        historic_retreat_df = retreat_df.sub(
            (all_merged["year"] - custom_ref_year) * all_merged["trend"].round(2), axis=0
        )
        merged_retreat_df = pd.concat([all_merged, historic_retreat_df], axis=1)

    return merged_retreat_df

def extend_transects_4_new_distances_points(subset, new_distances):
    """ Extend reference coastsat transects when schange value is bigger than transect length.
     Then determine new shoreline point. Returns new points locations, and extended transects. """

    # Convert orientation to radians
    orientations_rad = np.deg2rad(subset['orientation'])
    new_geoms = []
    new_transects = []

    # Interpolate points at new distances
    # new_points = subset.geom_transect_coastsat.to_crs(2193).interpolate(new_distances)

    # Iterate through rows
    for idx, row in subset.reset_index(drop=True).iterrows():
        line = row.geom_transect_coastsat
        if line is None or line.is_empty or not isinstance(line, LineString):
            new_geoms.append(None)
            new_transects.append(None)
            continue

        line = gpd.GeoSeries([line], crs=subset.crs).to_crs(2193).iloc[0]
        new_dist = new_distances.iloc[idx]
        orientation = orientations_rad.iloc[idx]

        if pd.isna(new_dist):
            new_geoms.append(None)
            new_transects.append(None)
            continue

        if 0 <= new_dist <= line.length:
            # Interpolate as normal
            new_point = line.interpolate(new_dist)
        else:
            # Extrapolation needed
            # Get start or end point depending on whether new_dist is <0 or >length
            if new_dist < 0:
                base_point = Point(line.coords[0])
                distance_out = -new_dist
                # Move backward (orientation + 180)
                theta = orientation + np.pi
                # Extend at start
            else:
                base_point = Point(line.coords[-1])
                distance_out = new_dist - line.length
                # Move forward
                theta = orientation
                # Extend at end             
            # Compute new point using simple trigonometry
            dx = distance_out * np.sin(theta)
            dy = distance_out * np.cos(theta)
            new_point = Point(base_point.x + dx, base_point.y + dy)

        # ---- build new transect AFTER new_point is finalized ----
        coords = list(line.coords)
        if new_dist < 0:
            new_line = LineString([new_point] + coords)
        elif new_dist > line.length:
            new_line = LineString(coords + [new_point])
        else:
            new_line = line  # unchanged

        new_geoms.append(new_point)
        new_transects.append(new_line)

    return new_geoms, new_transects

def points_to_polylines(subset):
    """ Take the points GeoDataFrame and transforms to polylines"""
    missing = subset[subset.geometry.isnull()]
    print(f"Missing geometries: {len(missing)}")

    subset = subset.dropna(subset=['geom_new_points'])

    # Step 1: Extract group and order fields
    subset["group_id"] = subset["coastsat_transect_id"].str.split("-").str[0]
    subset["order_id"] = subset["coastsat_transect_id"].str.split("-").str[1].astype(int)

    # Step 2: Create LineStrings per group
    lines = []

    for group_id, group in subset.groupby("group_id"):
        sorted_group = group.sort_values(by="order_id")
        coords = sorted_group.geometry.tolist()
        
        # Ensure we have at least 2 points to make a line
        if len(coords) >= 2:
            line = LineString(coords)
            lines.append({"geometry": line, "group_id": group_id})

    # Step 3: Create GeoDataFrame of LineStrings
    lines_gdf = gpd.GeoDataFrame(lines, crs=subset.crs)

    # Step 4: Reproject to WGS84 (EPSG:4326) for web tools like Leaflet / Google Earth
    lines_gdf = lines_gdf.to_crs(epsg=4326)
    return(lines_gdf)
############################################################################################
#%%
#CRS_NZTM , CRS_WGS84
meta_data_fp = "NZ_VLM_final_May24.csv"
slr_fp = f"NZ_Searise_noVLM-2005_{custom_ref_year}adjusted.csv"
meta_data = load_metadata_nzrise(meta_data_fp, crs_str= CRS_WGS84 ) # gpd.DataFrame
slr_data  = load_slrdata_nzrise(slr_fp) #pd.DataFrame

merged= pd.merge(meta_data,slr_data,
                        on='nzrise_site_id', how='outer')
nzrise_merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{CRS_WGS84}", geometry= 'geom_nzrise')

coastsat_merged = load_and_merge_coastsat_data(
    "transects_reindexed.geojson",
    f"points_ref_shoreline_{custom_ref_year}.geojson",
    CRS_WGS84
)

#Calculate distances to nearest NZRise points
distances, nearest_indices = nearest_points(meta_data,
                                             coastsat_merged)

#Merge nzrise_merged & coastsat_merged, based on nearest_indices
#to each coastsat_transect_id there is a matching nzrise_site_id
#nzrise_site_id is repeated, repeat also coastsat_transect_id as many times needed
#add column of nzrise_site_id to coastsat_merged

coastsat_merged["nzrise_site_id"]=meta_data["nzrise_site_id"].iloc[nearest_indices].values

#Now merge based on nzrise_site_id
all_merged = pd.merge( coastsat_merged, nzrise_merged,
                      on='nzrise_site_id', how='outer')
#Print all columns, there should be 3 geometry columns (points ref , coastsat transects
# and nzrise points).

print(all_merged.columns)
#Transform into geopandas? A geometry column needs to be picked, points ref .

all_merged = gpd.GeoDataFrame(all_merged,crs=f"EPSG:{CRS_WGS84}",
                               geometry= 'geom_points_ref')


#%%
#Step back, only Kaipara
# merged_kaipara= all_merged[all_merged.coastsat_site_id == 'nzd0126']

#Now calc retreat
# retreat = calc_retreat(merged_kaipara)

# all_merged['beach_slope'] = all_merged.groupby('coastsat_site_id')['beach_slope']\
#                                     .transform(fillna_mildslop_smooth)

# zero_sites = all_merged.groupby('coastsat_site_id')['beach_slope'] \
#                        .apply(lambda x: (x == 0).all())
# print(zero_sites[zero_sites].index.tolist())

# all_merged('coastsat_site_id')

retreat = calc_retreat(all_merged, retreat_scen_marker)


#%%
#MWE of retreat polyline in one site
# Get unique combinations

years =  [2025, 2100]
scenarios = [1.9]
# years =  [2005, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
# unique_scenarios = [1.9,2.6,4.5,7,8.5]


available_years = sorted(retreat["year"].unique())
upper_bound = min([y for y in available_years if y >= custom_ref_year])

# Quantiles to process
slr_qt_list = ["17_shifted","50_shifted","83_shifted"]
# slr_qt_list = ["17", "50", "83"]

for slr_qt in slr_qt_list:
    print(f"\n=== Processing SLR quantile: {slr_qt} ===")
        
    for year in years: 
        for scenario in scenarios:
            start_time = time.time()  # ⏱ start timer
            print(f"Processing scenario {scenario} for year {year} (quantile {slr_qt})...")

            #If year base, don't extend transects, append points as they are
            if year == custom_ref_year:

                #Since we have to create data for the ref year (unless already in available years)
                #Use next available year  to copy geometries, then set all retreats_shifted to zero

                subset = retreat[(retreat['year'] == upper_bound) & (retreat['scenario'] == scenario)]
                # set all retreats quartiles to zero

                subset = subset.reset_index(drop=True)
                subset['geom_new_points'] = subset["geom_points_ref"]
                #Just pass transects as they are
                subset["extended_transects"] =  subset["geom_transect_coastsat"]

                #Set to zero
                slr_quantiles_shift = ["17_shifted", "50_shifted", "83_shifted"]

                for q in slr_quantiles_shift:
                    subset[f"retreat_{q}"]   = 0

            else:
            
                # Subset dataframe with specific projection year & specific scenario
                subset = retreat[(retreat['year'] == year) & (retreat['scenario'] == scenario)]
                #Scenarios SSP2-2.6 & SSP5-8.5 have duplicates
                subset = subset.drop_duplicates(subset='coastsat_transect_id', keep='last')

                #Calc new point location according to retreat_50
                bruun_slr_qt= subset[f"retreat_{slr_qt}"]             #projected retreat 50 quantile

                subset.geom_transect_coastsat.to_crs(2193).length   #transects lengths
                # Reproject to NZTM2000 (meters)

                # Project reference point onto the transect line (get distance along the line)
                ref_distances_along_lines = subset.geom_transect_coastsat.to_crs(2193) .project(
                                            subset.geom_points_ref.to_crs(2193)  )

                # Calculate new distance from start of line
                new_distances = ref_distances_along_lines - bruun_slr_qt
                
                #Determine new points for shoreline , extend transects when needed
                new_geoms, new_transects = extend_transects_4_new_distances_points(subset, new_distances)
                # Create a GeoSeries and convert back to WGS84
                new_points = gpd.GeoSeries(new_geoms, crs=2193).to_crs(4326).reset_index(drop=True)
                # Make GeoSeries of new transects in WGS84
                new_transects = gpd.GeoSeries(new_transects, crs=2193).to_crs(4326).reset_index(drop=True)

                # Also reset subset to ensure 1-to-1 alignment
                subset = subset.reset_index(drop=True)

                # Assign to DataFrame
                subset['geom_new_points'] = new_points#.to_crs(4326)
                subset["extended_transects"] = new_transects#.to_crs(4326)

            #% From points to linestrings, careful on what active geometry goes inside
            lines_gdf = points_to_polylines(subset.set_geometry('geom_new_points'))
            # lines_gdf = points_to_polylines(merged_df.set_geometry('geom_smoothed_new_points'))
            lines_gdf.to_file(f"{retreat_scen_marker}_lines_shoreline_{slr_qt}qtl_{year}_{scenario}.geojson")
            cols_to_display= ['geom_new_points',f'{slr_qt}',f'retreat_{slr_qt}']
            # cols_to_display= ['geom_smoothed_new_points','50','retreat_50']
            subset[cols_to_display].to_file(f"{retreat_scen_marker}_points_shoreline_{slr_qt}qtl_{year}_{scenario}.geojson")

            elapsed_minutes = (time.time() - start_time) / 60
            print(f"⏱ Time for scenario {scenario} ({year}, {slr_qt} qtl): {elapsed_minutes:.2f} minutes")
            print(f"Scenario {scenario}, projections for year {year}, quantile {slr_qt}, saved!\n")

            #Save for postprocessing
            subset.to_pickle(f"./postprocessing/{retreat_scen_marker}_scenario{scenario}_year{year}_quantile_{slr_qt}.pkl")
            
# %%
