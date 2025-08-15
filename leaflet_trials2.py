#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:22:51 2025

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
<html>
<head>
  <title>Retreat Scenarios</title>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  
  <!-- Leaflet CSS -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    #map { height: 100vh; }
  </style>
</head>
<body>

<div id="map"></div>

<!-- Leaflet JS -->
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<script>
  var map = L.map('map').setView([0, 0], 2);

  // Base map
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    attribution: '© OpenStreetMap contributors'
  }).addTo(map);

  // Store each individual layer for fine control
  var layers = {
    "Scenario 1.9 - 2005": L.layerGroup(),
    "Scenario 1.9 - 2020": L.layerGroup(),
    "Scenario 2.6 - 2005": L.layerGroup(),
    "Scenario 2.6 - 2020": L.layerGroup()
  };

  // Define a color scale for retreat_50
  function getColor(value) {
    return value > 50 ? '#800026' :
           value > 40 ? '#BD0026' :
           value > 30 ? '#E31A1C' :
           value > 20 ? '#FC4E2A' :
           value > 10 ? '#FD8D3C' :
           value > 5  ? '#FEB24C' :
           value > 0  ? '#FED976' :
                        '#FFEDA0';
  }

  // Convert retreat_50 to marker size
  function getRadius(value) {
    return value > 0 ? Math.sqrt(value) * 2 : 2;
  }

  // Popup
  function onEachFeature(feature, layer) {
    if (feature.properties && feature.properties.retreat_50 !== undefined) {
      layer.bindPopup(`Retreat: ${feature.properties.retreat_50}`);
    }
  }

  // Style function for each point
  function pointToLayer(feature, latlng) {
    const value = feature.properties.retreat_50;
    return L.circleMarker(latlng, {
      radius: getRadius(value),
      fillColor: getColor(value),
      color: "#000",
      weight: 1,
      opacity: 1,
      fillOpacity: 0.8
    });
  }

  // Load GeoJSON files with metadata
  var files = [
    {file: 'retreat_1.9_2005_50percentile.geojson', key: "Scenario 1.9 - 2005"},
    {file: 'retreat_1.9_2020_50percentile.geojson', key: "Scenario 1.9 - 2020"},
    {file: 'retreat_2.6_2005_50percentile.geojson', key: "Scenario 2.6 - 2005"},
    {file: 'retreat_2.6_2020_50percentile.geojson', key: "Scenario 2.6 - 2020"}
  ];

  let bounds = [];

  files.forEach(entry => {
    fetch(entry.file)
      .then(response => response.json())
      .then(data => {
        const geojson = L.geoJSON(data, {
          pointToLayer: pointToLayer,
          onEachFeature: function(feature, layer) {
            onEachFeature(feature, layer);
            if (layer.getLatLng) {
              bounds.push(layer.getLatLng());
            }
          }
        });
        geojson.addTo(layers[entry.key]);
      })
      .catch(err => console.error("Error loading " + entry.file, err));
  });

  // Add all layers to the map
  Object.values(layers).forEach(layer => map.addLayer(layer));

  // Add layer control
  L.control.layers(null, layers, {collapsed: false}).addTo(map);

  // Zoom to all data once all layers are loaded
  setTimeout(() => {
    if (bounds.length > 0) {
      map.fitBounds(L.latLngBounds(bounds));
    }
  }, 1000); // Allow a second for layers to load

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