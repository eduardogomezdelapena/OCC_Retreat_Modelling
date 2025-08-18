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
  <title>CoastSat Transects - SLR Layers</title>
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
      font-size: 14px;
    }
    .custom-control label {
      display: block;
      margin-left: 10px;
    }
    .custom-control .group {
      margin-bottom: 10px;
    }
  </style>
</head>
<body>
  <div id="map"></div>

  <!-- Custom toggle control -->
  <div class="custom-control">
    <div class="group">
      <strong>SLR 1.9</strong>
      <label><input type="checkbox" id="layer19_2005"> 2005</label>
      <label><input type="checkbox" id="layer19_2020"> 2020</label>
    </div>
    <div class="group">
      <strong>SLR 2.6</strong>
      <label><input type="checkbox" id="layer26_2005"> 2005</label>
      <label><input type="checkbox" id="layer26_2020"> 2020</label>
    </div>
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

    // Sublayers
    const slr19_2005 = L.layerGroup();
    const slr19_2020 = L.layerGroup();
    const slr26_2005 = L.layerGroup();
    const slr26_2020 = L.layerGroup();

    // Parent groups
    const slr19Group = L.layerGroup([slr19_2005, slr19_2020]);
    const slr26Group = L.layerGroup([slr26_2005, slr26_2020]);

    // Load GeoJSON and populate layer group
    function loadGeoJsonToLayer(url, layerGroup, fit = false) {
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

          if (fit && !map._hasFitBounds) {
            map.fitBounds(geoLayer.getBounds());
            map._hasFitBounds = true;
          }
        });
    }

    // Load all files into their layer groups (do not add to map yet)
    loadGeoJsonToLayer('retreat_1.9_2005_50percentile.geojson', slr19_2005, true);
    loadGeoJsonToLayer('retreat_1.9_2020_50percentile.geojson', slr19_2020);
    loadGeoJsonToLayer('retreat_2.6_2005_50percentile.geojson', slr26_2005);
    loadGeoJsonToLayer('retreat_2.6_2020_50percentile.geojson', slr26_2020);

    // Toggle handler
    function setupCheckbox(id, parentGroup, sublayer) {
      document.getElementById(id).addEventListener('change', function(e) {
        if (e.target.checked) {
          parentGroup.addLayer(sublayer);
          if (!map.hasLayer(parentGroup)) {
            parentGroup.addTo(map);
          }
        } else {
          parentGroup.removeLayer(sublayer);
        }
      });
    }

    // Setup checkboxes
    setupCheckbox('layer19_2005', slr19Group, slr19_2005);
    setupCheckbox('layer19_2020', slr19Group, slr19_2020);
    setupCheckbox('layer26_2005', slr26Group, slr26_2005);
    setupCheckbox('layer26_2020', slr26Group, slr26_2020);
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