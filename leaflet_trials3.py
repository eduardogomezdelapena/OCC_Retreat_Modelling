#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 11:15:07 2025

@author: egom802
"""

#%% HTML - Leaflet
import os
url_sv_gj="/home/egom802/Documents/GitHub/OCC_Retreat_Modelling/"
os.chdir(url_sv_gj)
'shoreline_retreat.geojson'

import json

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CoastSat Transects - SLR 1.9</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Leaflet -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <style>
    html, body, #map {
      height: 100%;
      margin: 0;
    }
    .custom-control {
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 1000;
      background: white;
      padding: 10px;
      border-radius: 5px;
      box-shadow: 0 0 5px rgba(0,0,0,0.3);
      font-family: sans-serif;
    }
    .custom-control label {
      display: block;
      margin-bottom: 5px;
    }
  </style>
</head>
<body>
  <div id="map"></div>

  <!-- Custom toggle control -->
  <div class="custom-control">
    <strong>SLR 1.9</strong><br>
    <label><input type="checkbox" id="layer2005" checked> 2005</label>
    <label><input type="checkbox" id="layer2020" checked> 2020</label>
  </div>

  <script>
    // Initialize map
    const map = L.map('map').setView([-42, 172], 6);

    // Basemap
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    // Color + radius
    function getColor(value) {
      return value > 10 ? '#d73027' :
             value > 5  ? '#fee08b' :
                          '#1a9850';
    }
    function getRadius(value) {
      return Math.max(4, value * 0.8);
    }

    // Layer groups for sublayers
    const sublayer2005 = L.layerGroup();
    const sublayer2020 = L.layerGroup();

    // Master group (parent layer)
    const slr19Group = L.layerGroup([sublayer2005, sublayer2020]).addTo(map);

    // Load function
    function loadGeoJsonToLayer(url, layerGroup) {
      fetch(url)
        .then(res => res.json())
        .then(data => {
          const geoLayer = L.geoJSON(data, {
            pointToLayer: function (feature, latlng) {
              const retreat = feature.properties.retreat_50;
              return L.circleMarker(latlng, {
                radius: getRadius(retreat),
                fillColor: getColor(retreat),
                color: '#000',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
              });
            },
            onEachFeature: function (feature, layer) {
              const name = feature.properties.name;
              const retreat = feature.properties.retreat_50;
              layer.bindPopup(`<strong>${name}</strong><br>Retreat (50th percentile): ${retreat} m`);
            }
          });
          geoLayer.addTo(layerGroup);

          // Fit only once
          if (!map._hasFitBounds) {
            map.fitBounds(geoLayer.getBounds());
            map._hasFitBounds = true;
          }
        });
    }

    // Load files into sublayers
    loadGeoJsonToLayer('retreat_1.9_2005_50percentile.geojson', sublayer2005);
    loadGeoJsonToLayer('retreat_1.9_2020_50percentile.geojson', sublayer2020);

    // Toggle logic
    document.getElementById('layer2005').addEventListener('change', function(e) {
      if (e.target.checked) {
        slr19Group.addLayer(sublayer2005);
      } else {
        slr19Group.removeLayer(sublayer2005);
      }
    });

    document.getElementById('layer2020').addEventListener('change', function(e) {
      if (e.target.checked) {
        slr19Group.addLayer(sublayer2020);
      } else {
        slr19Group.removeLayer(sublayer2020);
      }
    });
  </script>
</body>
</html>

"""
# Save HTML file
with open("index.html", "w") as f:
    f.write(html_content)

print("✅ map.html and data.geojson generated. Open map.html in browser (preferably via a local server).")
#Run in bash
"python -m http.server"
"http://localhost:8000/index.html"