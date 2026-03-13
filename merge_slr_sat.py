""" script with functions to create all_merged:
the joint sea level rise dataset and the coastsat data"""

#%%
import pandas as pd
import geopandas as gpd
from utils import load_metadata_nzrise, load_slrdata_nzrise, load_and_merge_coastsat_data, nearest_points
#%%

# Constants
CRS_WGS84 = 4326 # Lat/Lon
custom_ref_year = 2025


meta_data_fp = "NZ_VLM_final_May24.csv"
slr_fp = f"NZ_Searise_noVLM-2005_{custom_ref_year}adjusted.csv"
meta_data = load_metadata_nzrise(meta_data_fp, crs_str= CRS_WGS84 ) # gpd.DataFrame
slr_data  = load_slrdata_nzrise(slr_fp) #pd.DataFrame

merged= pd.merge(meta_data,slr_data,
                        on='nzrise_site_id', how='outer')
nzrise_merged = gpd.GeoDataFrame(merged, crs=f"EPSG:{CRS_WGS84}", geometry= 'geom_nzrise')

coastsat_merged = load_and_merge_coastsat_data(
    "transects_reindexed_Nickupdate.geojson",
    f"points_ref_shoreline_{custom_ref_year}_Nickupdate.geojson",
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
                      on='nzrise_site_id', how='left')
#Print all columns, there should be 3 geometry columns (points ref , coastsat transects
# and nzrise points).

print(all_merged.columns)
#Transform into geopandas? A geometry column needs to be picked, points ref .

all_merged = gpd.GeoDataFrame(all_merged,crs=f"EPSG:{CRS_WGS84}",
                               geometry= 'geom_points_ref')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# define output directory and file paths
OUTPUT_DIR = "outputs"
ALL_MERGED_FP = f"{OUTPUT_DIR}/all_merged.gpkg"

# helper functions for saving

def save_all_merged(df: gpd.GeoDataFrame, path: str = ALL_MERGED_FP) -> None:
    """Persist *all_merged* to disk.

    This routine ensures only a single geometry column is active, converting
    any additional geometry columns to WKT strings so that GeoPackage (and
    other file formats) can be written without error.

    Parameters
    ----------
    df : GeoDataFrame
        The merged dataset produced by this script.  It may contain multiple
        geometry columns; the active ``.geometry`` attribute will be used for
        output while the others are converted to text.
    path : str
        Destination filename (must include directory).  Defaults to the
        GeoPackage under ``outputs/``.
    """
    gdf = df.copy()

    # make sure the geometry column is set to the one we intend to write
    if gdf.geometry.name != 'geom_points_ref':
        if 'geom_points_ref' in gdf.columns:
            gdf = gdf.set_geometry('geom_points_ref')

    # convert any other geometry columns to WKT to avoid the multiple-geometry error
    for col in gdf.columns:
        # older geopandas releases don't expose `GeoDtype` at the top level; use
        # the array submodule instead which has a `GeometryDtype` class that
        # behaves identically.  
        if col != gdf.geometry.name and isinstance(
            gdf[col].dtype,
            gpd.array.GeometryDtype,
        ):
            gdf[col] = gdf[col].apply(lambda geom: geom.wkt if geom is not None else None)

    gdf.to_file(path, layer="all", driver="GPKG")


# save the dataframe so other scripts can consume it later
save_all_merged(all_merged)  # writes to outputs/all_merged.gpkg

# %%
