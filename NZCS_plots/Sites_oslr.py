

#%%
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import box, Point

#%%
def load_lines(filepath: str):
    """Load ref points (shoreline) and reproject, drop missing geometries.
    Points externally generated in gen_ref.py"""
    points = gpd.read_file(filepath)
    return points.dropna(subset=["geometry"])

def clip_to_christchurch(gdf):
    """Clip GeoDataFrame to Christchurch bounding box."""
    # Approximate bounding box around Christchurch (in WGS84)
    xmin, xmax = 172.4, 173.0
    ymin, ymax = -43.8, -43.3
    chch_bbox = box(xmin, ymin, xmax, ymax)
    bbox_gdf = gpd.GeoDataFrame(geometry=[chch_bbox], crs="EPSG:4326")

    # Ensure same CRS before clipping
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    # Clip to Christchurch area
    clipped = gpd.clip(gdf, bbox_gdf)
    return clipped


def point_plot(shoreline_ref, custom_ref_year):
    """Plots extracted shoreline and Christchurch city."""
    shoreline_web_mercator = shoreline_ref.to_crs(epsg=3857)

    # Christchurch city location (approx. city centre)
    chch_point = gpd.GeoDataFrame(
        geometry=[Point(172.6362, -43.5321)], crs="EPSG:4326"
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot shoreline
    shoreline_web_mercator.plot(ax=ax, color='red', linewidth=2, label=f"{custom_ref_year} Shoreline")

    # Plot Christchurch marker
    chch_point.plot(ax=ax, color='yellow', markersize=80, edgecolor='black', label="Christchurch City")

    # Add basemap showing the city layout
    ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldImagery, zoom=12)

    ax.set_title(f"Christchurch Shoreline and City ({custom_ref_year})", fontsize=14)
    ax.legend()
    ax.set_axis_off()
    plt.show()


#% Parameters
custom_ref_year = 2025
CRS_NZTM = 2193  # NZ Transverse Mercator
CRS_WGS84 = 4326 # Lat/Lon


# # --- 1. Load the GeoJSON shoreline file ---
# bs= '/home/egom802/Documents/GitHub/OCC_Retreat_Modelling/'
# shoreline_fp = bs + "htrend_lines_shoreline_50_shiftedqtl_2100_8.5.geojson"
# shorelines = gpd.read_file(shoreline_fp)

#%
lines = load_lines(f"../htrend_lines_shoreline_50_shiftedqtl_{custom_ref_year}_1.9.geojson")
lines_chch = clip_to_christchurch(lines)
point_plot(lines_chch, custom_ref_year)
# %%
