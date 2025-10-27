
#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
#%%
def load_points(filepath: str):
    """Load ref points (shoreline) and reproject, drop missing geometries.
    Points externally generated in gen_ref.py"""
    points = gpd.read_file(filepath)
    return points.dropna(subset=["geometry"])

def point_plot(shoreline_ref, custom_ref_year):
    """ Plots extracted shoreline and reference transects"""
    # Convert data to Web Mercator (EPSG:3857) for plotting with basemap
    shoreline_web_mercator = shoreline_ref.to_crs(epsg=3857)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    # Plot shoreline and transects in the correct projection
    shoreline_web_mercator.plot(ax=ax, color='red', markersize=5, label="2005 Shoreline Points")

    # Add the basemap
    ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldImagery)
    # Customize the plot
    ax.set_title(f" {custom_ref_year} Shoreline Points Across All NZ Sites")
    ax.legend()
    ax.set_axis_off()
    # Show the map
    plt.show()

def multipolygon_plot(multipolygon_df):
    """ Plots rivers multipolygon"""

    # Convert data to Web Mercator (EPSG:3857) for plotting with basemap
    multipolygon_web_mercator = multipolygon_df.to_crs(epsg=3857)

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12))
    # Plot shoreline and transects in the correct projection
    multipolygon_web_mercator.plot(ax=ax, color='lightblue', edgecolor='red')

    # Add the basemap
    ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldImagery)
    # Customize the plot
    ax.set_title(f"Rivers Across All NZ")
    ax.legend()
    ax.set_axis_off()
    # Show the map
    plt.show()

#%% Parameters
custom_ref_year = 2025
CRS_NZTM = 2193  # NZ Transverse Mercator
CRS_WGS84 = 4326 # Lat/Lon

#%%
points = load_points(f"points_ref_shoreline_{custom_ref_year}.geojson")
point_plot(points, custom_ref_year)
#%% Load NZ river mouths

rivers= gpd.read_file("./preprocessing/nz-river-polygons-topo-150k.gpkg")
rivers= gpd.GeoDataFrame(rivers, geometry="geometry")
multipolygon_plot(rivers)

# %%

#Make sure they are in the same crs
points = points.to_crs(CRS_NZTM )
rivers = rivers.to_crs(CRS_NZTM )

# Create a 10 km (10,000 m) buffer around polygons
buffered = rivers.copy()
buffered["geometry"] = buffered.buffer(10_000)
# %%
# Step 2. Spatial join to find points within 10 km of polygons
joined = gpd.sjoin(points, buffered, predicate="intersects", how="left")

# Step 3. Keep only points NOT within 10 km
points_far = joined[joined["index_right"].isna()].drop(columns="index_right")

# Optional: reset index
points_far = points_far.reset_index(drop=True)
# %%
point_plot(points, custom_ref_year)
multipolygon_plot(rivers)
point_plot(points_far, custom_ref_year)
# %%

print(points.crs)
print(rivers.crs)

# %%
points.geometry.iloc[0].distance(points.geometry.iloc[1])
# %%
