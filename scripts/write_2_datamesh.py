""" Write data to Oceanum's DataMesh. Quick trial with .geojson files."""

#%%

import os
import pandas as pd
import geopandas as gpd
import numpy as np
from oceanum.datamesh import Connector

#DataMesh token
datamesh=Connector(token="3052e2bdd10904ae353ac54ed205df32bfcc20e2")

#%%

# Read the baseline.geojson file 
gdf_oslr_base = gpd.read_file("oslr_lines_shoreline_50_shiftedqtl_2025_1.9.geojson")

#Read  .geojson files for future scenarios and append each scenario to the same geodataframe
gdf_oslr_2100_1_9 = gpd.read_file("oslr_lines_shoreline_50_shiftedqtl_2100_1.9.geojson")        
gdf_oslr_2100_2_6 = gpd.read_file("oslr_lines_shoreline_50_shiftedqtl_2100_2.6.geojson")
gdf_oslr_2100_4_5 = gpd.read_file("oslr_lines_shoreline_50_shiftedqtl_2100_4.5.geojson")
gdf_oslr_2100_7 = gpd.read_file("oslr_lines_shoreline_50_shiftedqtl_2100_7.geojson")
gdf_oslr_2100_8_5 = gpd.read_file("oslr_lines_shoreline_50_shiftedqtl_2100_8.5.geojson")    

#Before concatenating, add a column to each future scenario geodataframe to indicate the RCP scenario
gdf_oslr_2100_1_9['rcp'] = '1.9'
gdf_oslr_2100_2_6['rcp'] = '2.6'
gdf_oslr_2100_4_5['rcp'] = '4.5'
gdf_oslr_2100_7['rcp'] = '7'
gdf_oslr_2100_8_5['rcp'] = '8.5'

#Now merge all the future scenarios into one geodataframe, with a column for the scenario name
gdf_oslr_2100 = pd.concat([gdf_oslr_2100_1_9, gdf_oslr_2100_2_6, gdf_oslr_2100_4_5, gdf_oslr_2100_7, gdf_oslr_2100_8_5], ignore_index=True)
gdf_oslr_2100['scenario'] = gdf_oslr_2100.apply(lambda row: f"RCP {row['rcp']}", axis=1)


# %%
datasource = datamesh.write_datasource(datasource_id='occ_shorelinechange_base', 
                          name="Shoreline baseline reference (2025)",
                          description="Linestrings representing the average shoreline position in 2025 (baseline year). Average shoreline position of satellite images for 2024-2025 using CoastsatNZ.", 
                          data=gdf_oslr_base                              
                         )

#%%

datasource_2100 = datamesh.write_datasource(datasource_id='occ_shorelinechange_oslr_50p_2100', 
                          name="Only sea-level rise future scenarios (2100)",
                          description="Linestrings representing the average shoreline position in 2100 for different RCP scenarios. 50th percentile.", 
                          data=gdf_oslr_2100                              
                         )

# %%

#Repeat for the scenario with historic trends
gdf_htrend_base = gpd.read_file("htrend_lines_shoreline_50_shiftedqtl_2025_1.9.geojson")

#Read  .geojson files for future scenarios and append each scenario to the same geodataframe
gdf_htrend_2100_1_9 = gpd.read_file("htrend_lines_shoreline_50_shiftedqtl_2100_1.9.geojson")        
gdf_htrend_2100_2_6 = gpd.read_file("htrend_lines_shoreline_50_shiftedqtl_2100_2.6.geojson")
gdf_htrend_2100_4_5 = gpd.read_file("htrend_lines_shoreline_50_shiftedqtl_2100_4.5.geojson")
gdf_htrend_2100_7 = gpd.read_file("htrend_lines_shoreline_50_shiftedqtl_2100_7.geojson")
gdf_htrend_2100_8_5 = gpd.read_file("htrend_lines_shoreline_50_shiftedqtl_2100_8.5.geojson")  

#Before concatenating, add a column to each future scenario geodataframe to indicate the RCP scenario
gdf_htrend_2100_1_9['rcp'] = '1.9'
gdf_htrend_2100_2_6['rcp'] = '2.6'
gdf_htrend_2100_4_5['rcp'] = '4.5'
gdf_htrend_2100_7['rcp'] = '7'              
gdf_htrend_2100_8_5['rcp'] = '8.5'

#Now merge all the future scenarios into one geodataframe, with a column for the scenario name
gdf_htrend_2100 = pd.concat([gdf_htrend_2100_1_9, gdf_htrend_2100_2_6, gdf_htrend_2100_4_5, gdf_htrend_2100_7, gdf_htrend_2100_8_5], ignore_index=True)
gdf_htrend_2100['scenario'] = gdf_htrend_2100.apply(lambda row: f"RCP {row['rcp']}", axis=1)    

datasource_htrend_2100 = datamesh.write_datasource(datasource_id='occ_shorelinechange_htrend_50p_2100', 
                          name="Historic trend + sea-level rise future scenarios (2100)",
                          description="Linestrings representing the average shoreline position in 2100 for different RCP scenarios, including the effect of historic shoreline change trends. 50th percentile.", 
                          data=gdf_htrend_2100                              
                         )  




#Delete a DataSource from the DataMesh (if needed)
#datamesh.delete_datasource("occ_shorelinechange_oslr_base")


# %%
