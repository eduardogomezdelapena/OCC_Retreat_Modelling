
import geopandas as gpd
import pandas as pd
import math
from shapely.geometry import Point, LineString, Polygon
from shapely import line_interpolate_point
import contextily as ctx
ctx.set_cache_dir("contextily_cache")
pd.set_option("display.max_columns", None)

#%%
bruun = gpd.read_file("https://github.com/eduardogomezdelapena/OCC_Retreat_Modelling/raw/refs/heads/main/retreat_8.5_2100_50percentile.geojson")

#%%
points_2005= gpd.read_file('/home/egom802/Documents/GitHub/OCC_Retreat_Modelling/2005_ref_points.geojson')
#%%
transects = gpd.read_file("https://uoa-eresearch.github.io/CoastSat/transects_extended.geojson")



