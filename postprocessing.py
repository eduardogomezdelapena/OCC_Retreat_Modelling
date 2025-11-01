
""" Post-processing """
#%%
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import geodatasets

#%%
year =  [2100]
scenarios = [4.5,8.5]
slr_qt= ["50"]
scn_strs = ['oslr','htrend']

for scenario in scenarios: 
    for scn_str in scn_strs: 

        #Load only Bruun 1.9 scenario, 2100, 50th percentile
        retreat = pd.read_pickle(f"./postprocessing/{scn_str}_scenario{scenario}_year{year[0]}_quantile_{slr_qt[0]}_shifted.pkl")
        retreat = gpd.GeoDataFrame(retreat, geometry="geom_new_points", crs="EPSG:4326")

        #%
        # Load Natural Earth data

        nz_outline= gpd.read_file("./postprocessing/regions/nz-coastlines-topo-150k.gpkg")
        nz_outline = gpd.GeoDataFrame(nz_outline, geometry="geometry")
        nz_outline = nz_outline.to_crs(retreat.crs)

        #  Load regions
        regions = gpd.read_file("./postprocessing/regions/regional-council-2025.gpkg")
        regions= gpd.GeoDataFrame(regions, geometry="geometry")
        #Make sure they are in the same crs
        regions = regions.to_crs(retreat.crs)

        #Retreat by region
        retreat_regions = gpd.sjoin(retreat, regions, how="left", predicate="within")

        regional_summary = (
            retreat_regions
            .groupby("REGC2025_V1_00_NAME")["retreat_50_shifted"]
            .median()
            .reset_index()
        )
        regions_summary = regions.merge(regional_summary, on="REGC2025_V1_00_NAME", how="left")

        # % Plot
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 10))
        regions_summary.plot(
            column="retreat_50_shifted",
            cmap="coolwarm",
            legend=True,
            edgecolor="black",
            linewidth=0.5,
            ax=ax,
            vmin=-45,   
            vmax=45     
        )
        nz_outline.plot(ax=ax, color="black", linewidth=0.6)

        if scn_str == 'oslr':
            scn_str_tit = "Only Sea-level rise"
        elif scn_str == 'htrend':
            scn_str_tit = 'SLR + Historic trend'

        if scenario == 1.9:
            scenario_tit = "SSP1—1.9 "
        elif scenario == 4.5:
            scenario_tit = 'SSP2—4.5'
        elif scenario == 8.5:
            scenario_tit = 'SSP5—8.5'

        ax.set_title(f"Median Coastal Change by Region (m): year 2100 \n {scn_str_tit}, {scenario_tit}", fontsize=14)
        ax.axis("off")
         
        print(f'Min {scenario_tit}, {scn_str_tit}: {regions_summary.retreat_50_shifted.min()}')
        print(f'Max {scenario_tit}, {scn_str_tit}: {regions_summary.retreat_50_shifted.max()}')

        output_path = f'./postprocessing/figs/{scn_str}_scenario{scenario}_year{year[0]}_quantile_{slr_qt[0]}_map.png'
        plt.savefig(output_path, bbox_inches="tight", transparent = True)
        # %%
