#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 09:51:59 2025

@author: egom802
"""

import json
from shapely.geometry import LineString, mapping
from collections import defaultdict

# Load the GeoJSON file with Points
with open("/home/egom802/Documents/GitHub/OCC_Retreat_Modelling/retreat_1.9_2050_50percentile.geojson") as f:
    data = json.load(f)

# Group points by coastsat_id prefix (e.g., 'nzd0001')
groups = defaultdict(list)
for feature in data['features']:
    if feature['geometry']['type'] == 'Point':
        coastsat_id = feature['properties'].get('coastsat_id')
        if coastsat_id:
            # Split by '-' and take prefix
            group_id = coastsat_id.split('-')[0]
            groups[group_id].append((coastsat_id, feature['geometry']['coordinates']))
#%%
# Sort points within each group by coastsat_id to maintain order and create LineStrings
linestring_features = []
for group_id, points in groups.items():
    # Sort by full coastsat_id (lexical order will work if IDs are zero-padded)
    points.sort(key=lambda x: x[0])
    coords = [pt[1] for pt in points]
    
    # Create LineString if there are at least 2 points
    if len(coords) > 1:
        line = LineString(coords)
        feature = {
            "type": "Feature",
            "geometry": mapping(line),
            "properties": {"group_id": group_id}
        }
        linestring_features.append(feature)

# Create new GeoJSON FeatureCollection with all LineStrings
new_geojson = {
    "type": "FeatureCollection",
    "features": linestring_features
}
#%%
# Save to new GeoJSON file
with open("/home/egom802/Documents/GitHub/OCC_Retreat_Modelling/lines_retreat_1.9_2005_50percentile_by_group.geojson", 'w') as f:
    json.dump(new_geojson, f, indent=2)

print("Converted Points to LineStrings by group successfully!")