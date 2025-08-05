#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:06:05 2025

@author: egom802
"""

#%% HTML - Leaflet

url_sv_gj="/home/egom802/Documents/GitHub/OCC_Retreat_Modelling/"

import json

# Load GeoJSON string (can also be from a file)
with open(url_sv_gj+ 'shoreline_retreat.geojson', 'r') as f:
    geojson_data = json.load(f)

geojson_str = json.dumps(geojson_data)

# HTML template with embedded GeoJSON
html_template = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Leaflet Map</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>#map {{ height: 600px; }}</style>
</head>
<body>
  <h2>Leaflet Map with GeoJSON</h2>
  <div id="map"></div>
  <script>
    var map = L.map('map').setView([-40.9006, 174.8860], 5);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    var geojsonData = {geojson_str};

    L.geoJSON(geojsonData, {{
      onEachFeature: function (feature, layer) {{
        if (feature.properties && feature.properties.name) {{
          layer.bindPopup(feature.properties.name);
        }}
      }}
    }}).addTo(map);
  </script>
</body>
</html>
"""

# Save to file
with open(url_sv_gj+"map.html", "w") as f:
    f.write(html_template)