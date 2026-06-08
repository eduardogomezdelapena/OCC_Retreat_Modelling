"""" Run retreat scenarios using data prepared with the utils.py and merge_slr_sat.py functions.
This script is intended to be run after data preparation steps have been completed,
 and will use the merged dataset to run retreat scenarios."""
#%%
import geopandas as gpd
import time
from utils import extend_transects_4_new_distances_points, points_to_polylines
from utils import calc_retreat
from shapely import wkt
#%%

OUTPUT_DIR = "outputs"
ALL_MERGED_FP = f"{OUTPUT_DIR}/all_merged.gpkg"
retreat_scen_marker = "htrend"

def load_all_merged(path=ALL_MERGED_FP):
    # load the geopackage and rename the active geometry column to a more
    # descriptive name so downstream code doesn't get confused when working
    # with multiple geometry types.
    gdf = gpd.read_file(path)

    # change the name of the active geometry column
    # from the default 'geometry' to 'geom_points_ref'. this also updates the
    # column list accordingly.
    gdf = gdf.rename_geometry("geom_points_ref")

    # ensure any geometry-like columns stored as WKT are converted back to
    # shapely objects; only the active geometry is preserved by the
    # GeoPackage driver, so other columns come through as strings.

    for col in ["geom_transect_coastsat", "geom_points_ref", "geom_nzrise"]:
        if col in gdf.columns and gdf[col].dtype == object:
            gdf[col] = gdf[col].astype(str).apply(wkt.loads)

    return gdf

#Takes a while because its a big file.
all_merged = load_all_merged()


#%%

# Constants
CRS_WGS84 = 4326 # Lat/Lon
custom_ref_year = 2025

import numpy as np

#In case we want to select specific sites to test
# all_merged = all_merged[
#     ~all_merged['coastsat_site_id'].isin(['nzd0581', 'nzd0580'])
# ]

retreat = calc_retreat(all_merged, retreat_scen_marker)
#MWE of retreat polyline in one site
# Get unique combinations
#%%
retreat.set_geometry("geom_transect_coastsat")
retreat.set_geometry("geom_points_ref")


#%%
years =  [2025, 2100]
scenarios = [2.6]
# scenarios = [1.9,2.6,4.5,7,8.5]
# years =  [2005, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
# unique_scenarios = [1.9,2.6,4.5,7,8.5]


available_years = sorted(retreat["year"].unique())
upper_bound = min([y for y in available_years if y >= custom_ref_year])

# Quantiles to process
slr_qt_list = ["50_shifted"]
# slr_qt_list = ["17_shifted","50_shifted","83_shifted"]

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
            # subset.to_pickle(f"./postprocessing/{retreat_scen_marker}_scenario{scenario}_year{year}_quantile_{slr_qt}.pkl")
            
# %%
