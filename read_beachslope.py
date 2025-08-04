#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 17:05:34 2025

@author: eegp
"""




import geopandas as gpd


# Define the path to your GeoJSON file
file_path = 'https://raw.githubusercontent.com/UoA-eResearch/CoastSat/d61c2052a4a0b9ed9763c6fb89fa7cabdba034ff/transects_extended.geojson'

gdf = gpd.read_file(file_path)

bs = gdf.beach_slope
