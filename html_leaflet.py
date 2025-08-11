#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:06:05 2025

@author: egom802
"""

#%% HTML - Leaflet
import os
url_sv_gj="/home/eegp/Documents/GitHub/OCC_Retreat_Modelling/"
os.chdir(url_sv_gj)
'shoreline_retreat.geojson'

import json

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CoastSat transects (including NZ)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">

  <!-- Leaflet core -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <!-- Leaflet.draw -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css" />
  <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>

  <!-- Leaflet.sidebar -->
  <link href="https://cdn.jsdelivr.net/npm/leaflet-sidebar-v2@3.2.3/css/leaflet-sidebar.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/leaflet-sidebar-v2@3.2.3/js/leaflet-sidebar.min.js"></script>

  <!-- Leaflet providers (Carto, ESRI, etc.) -->
  <script src="https://unpkg.com/leaflet-providers@1.3.0/leaflet-providers.js"></script>

  <!-- Leaflet Hash -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet-hash/0.2.1/leaflet-hash.min.js"></script>

  <!-- Font Awesome (for sidebar icons) -->
  <link href="https://maxcdn.bootstrapcdn.com/font-awesome/4.1.0/css/font-awesome.min.css" rel="stylesheet">

  <style>
    html, body, #map {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
    }

    .legend {
      color: white;
      padding: 10px;
      background-color: rgba(0, 0, 0, 0.8);
      border-radius: 5px;
      max-width: 500px;
    }

    .legend i {
      width: 18px;
      height: 18px;
      float: left;
      margin-right: 8px;
      opacity: 0.7;
      clear: left;
    }

    .legend h4 {
      margin-top: 0px;
    }

    .link {
      text-decoration: underline;
      cursor: pointer;
    }

    .fa {
      line-height: inherit;
    }

  </style>
</head>

<body>

  <div id="map"></div>

  <!-- Sidebar -->
  <div id="sidebar" class="leaflet-sidebar collapsed">
    <div class="leaflet-sidebar-tabs">
      <ul role="tablist">
        <li><a href="#home" role="tab"><i class="fa fa-bars"></i></a></li>
        <li><a href="https://github.com/eduardogomezdelapena/OCC_Retreat_Modelling"><i class="fa fa-github"></i></a></li>
      </ul>
    </div>

    <div class="leaflet-sidebar-content">
      <div class="leaflet-sidebar-pane" id="home">
        <div id="attribution">

          <p> Shoreline retreat projections for Aotearoa New Zealand. Data processed by Eduardo Gomez-de la Pena. Retreat was estimated using the Bruun rule, see <a href="https://github.com/eduardogomezdelapena/OCC_Retreat_Modelling/blob/main/README.md">README</a> for more information.</p>
          <p> This work is part of  <a href="https://searise.nz/">Our Changing Coasts</a> MBIE project, funded secured by Dr. Giovanni Coco and Dr. Karin Bryan, The University of Auckland. </p>

          <p><a href="https://uoa-eresearch.github.io/CoastSat/">NZ shoreline data</a> processed by Nick Young using the <a href="https://github.com/kvos/CoastSat/"> CoastSat </a> toolbox and the NIWA Tide API. Shorelines corrected to MSL.</p>
          <p>Sea-Level Rise projections taken from the <a href="https://searise.takiwa.co/map/"> the NZ SeaRise</a> : Te Tai Pari O Aotearoa programme. </p>

        </div>
      </div>
    </div>
  </div>

  <script>
    // Initialize map
    var map = L.map('map', {
      center: [-42, 1],
      zoom: 6
    });

    var hash = new L.Hash(map);

    var sidebar = L.control.sidebar({
      autopan: false,
      closeButton: true,
      container: 'sidebar',
      position: 'right'
    }).addTo(map).open("home");

    // Drawing tools
    var drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    var drawControl = new L.Control.Draw({
      position: 'topleft',
      draw: {
        polyline: {
          shapeOptions: { color: '#f357a1' }
        },
        polygon: true,
        rectangle: true,
        marker: false,
        circle: false,
        circlemarker: false
      },
      edit: {
        featureGroup: drawnItems,
        remove: true
      }
    });
    map.addControl(drawControl);

    // Basemaps
    var baseMaps = {
      "ESRI Imagery": L.tileLayer.provider("Esri.WorldImagery").addTo(map),
      "ESRI Topo": L.tileLayer.provider("Esri.WorldTopoMap"),
      "CartoDB Positron": L.tileLayer.provider("CartoDB.Positron"),
      "CartoDB Dark": L.tileLayer.provider("CartoDB.DarkMatter"),
      "OpenStreetMap": L.tileLayer.provider("OpenStreetMap.Mapnik")
    };

    L.control.layers(baseMaps, null, { position: 'topleft' }).addTo(map);

    // Color scale
    function getColor(value) {
      return value > 10 ? '#d73027' :
             value > 5  ? '#fee08b' :
                          '#1a9850';
    }

    // Radius scale
    function getRadius(value) {
      return Math.max(4, value * 0.8);
    }

    // Load GeoJSON
    fetch('shoreline_retreat.geojson')
      .then(response => response.json())
      .then(data => {
        // Retreat layer 1
        var retreatLayer1 = L.geoJSON(data, {
          pointToLayer: function (feature, latlng) {
            var retreat = feature.properties.bruun_retreat1;
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
            var name = feature.properties.name || "Unknown";
            var retreat = feature.properties.bruun_retreat1 || "N/A";
            layer.bindPopup("<strong>" + name + "</strong><br>Bruun Retreat 1: " + retreat + " m");
          }
        });
    
        // Retreat layer 2
        var retreatLayer2 = L.geoJSON(data, {
          pointToLayer: function (feature, latlng) {
            var retreat = feature.properties.bruun_retreat2;
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
            var name = feature.properties.name || "Unknown";
            var retreat = feature.properties.bruun_retreat2 || "N/A";
            layer.bindPopup("<strong>" + name + "</strong><br>Bruun Retreat 2: " + retreat + " m");
          }
        });
    
        // Add one layer by default
        retreatLayer1.addTo(map);
    
        // Add to overlays control
        var overlayMaps = {
          "Bruun Retreat 1": retreatLayer1,
          "Bruun Retreat 2": retreatLayer2
        };
        L.control.layers(null, overlayMaps, { position: 'topleft', collapsed: false }).addTo(map);
    
        map.fitBounds(retreatLayer1.getBounds());
      });
    // Legend
    var legend = L.control({ position: 'bottomright' });
    legend.onAdd = function (map) {
      var div = L.DomUtil.create('div', 'legend');
      var grades = [0, 5, 10];
      var labels = [];
      var colors = ['#1a9850', '#fee08b', '#d73027'];

      div.innerHTML += '<b>Bruun Retreat (m)</b><br>';
      for (var i = 0; i < grades.length; i++) {
        var from = grades[i];
        var to = grades[i + 1];
        div.innerHTML +=
          '<i style="background:' + colors[i] + '"></i> ' +
          (to ? from + '–' + to : from + '+') + '<br>';
      }

      return div;
    };
    legend.addTo(map);
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
"http://localhost:8000/map.html"