
""" Post-processing """
#%%
import os
import geopandas as gpd
import numpy as np
import pandas as pd

#%%
year =  [2100]
scenario = [1.9]
slr_qt= ["50"]


#Load only Bruun 1.9 scenario, 2100, 50th percentile
retreat = pd.read_pickle(f"./postprocessing/scenario{scenario[0]}_year{year[0]}_quantile_{slr_qt[0]}.pkl")
retreat = gpd.GeoDataFrame(retreat, geometry="geom_new_points", crs="EPSG:4326")


#%%
#National average
#Projected erosion by NZ region

lol= retreat.retreat_50

retreat.retreat_50.describe()

retreat.retreat_50.median()

unique_sites = retreat["coastsat_site_id"].unique()

# %%

unique_transects = retreat["coastsat_transect_id"].unique()

trend= retreat.trend
#%% Load regions

regions = gpd.read_file("./postprocessing/regions/regional-council-2025.gpkg")
regions= gpd.GeoDataFrame(regions, geometry="geometry")
regions.plot()
#Make sure they are in the same crs
regions = regions.to_crs(retreat.crs)
# %%
#Retreat by region


retreat_regions = gpd.sjoin(retreat, regions, how="left", predicate="within")

# %%
regional_summary = (
    retreat_regions
    .groupby("REGC2025_V1_00_NAME")["retreat_50"]
    .median()
    .reset_index()
)

regions_summary = regions.merge(regional_summary, on="REGC2025_V1_00_NAME", how="left")

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 10))
regions_summary.plot(
    column="retreat_50",
    cmap="coolwarm",
    legend=True,
    edgecolor="black",
    linewidth=0.5,
    ax=ax
)
ax.set_title("Median Coastal Change by Region (m)", fontsize=14)
ax.axis("off")
plt.show()

# %%

import branca

values = regions_summary['retreat_50'].dropna()
colormap = branca.colormap.LinearColormap(
    colors=['blue','white','red'],
    vmin=values.min(), vmax=values.max()
).to_step(n=8)

m = folium.Map(location=[-41.0, 174.0], zoom_start=5)

def style(feature):
    v = feature['properties'].get('retreat_50')
    return {
        'fillColor': '#gray' if v is None else colormap(v),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7
    }

folium.GeoJson(
    regions_summary.__geo_interface__,
    style_function=style,
    tooltip=folium.GeoJsonTooltip(fields=["REGC2025_V1_00_NAME", "retreat_50"])
).add_to(m)

colormap.caption = "Mean shoreline change (m)"
colormap.add_to(m)

# display / save
m.save("./postprocessing/trial_SPP1_retreat_map.html")
# %%
