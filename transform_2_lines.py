#%% Transform from points to lines

import geopandas as gpd
from shapely.geometry import Point, LineString
from pyproj import Transformer
import math

# Load your GeoJSON file
gdf = gpd.read_file("/home/egom802/Documents/GitHub/OCC_Retreat_Modelling/retreat_1.9_2050_50percentile.geojson")

# Set projection (assuming WGS84 input)
gdf = gdf.set_crs("EPSG:4326")

# Reproject to a metric CRS (UTM or other local projection)
gdf = gdf.to_crs(epsg=3857)  # Web Mercator (meters); consider using appropriate local UTM zone
gdf = gdf.dropna()

gdf.geometry()
#%%
def create_retreat_line(point, distance, angle_deg=90):
    """
    Create a LineString from the original point to a shifted point.
    - distance in meters
    - angle_deg: direction to shift (90 = north by default)
    """
    x0, y0 = point.x, point.y
    angle_rad = math.radians(angle_deg)
    
    # Compute new point shifted by 'distance' in the specified angle
    x1 = x0 + distance * math.cos(angle_rad)
    y1 = y0 + distance * math.sin(angle_rad)
    
    return LineString([(x0, y0), (x1, y1)])

# Create new geometry column with LineStrings
gdf['geometry'] = gdf.apply(
    lambda row: create_retreat_line(row.geometry, row['retreat_50'], angle_deg=90),  # Adjust angle as needed
    axis=1
)

# Optionally reproject back to WGS84
gdf = gdf.to_crs(epsg=4326)

# Save to new GeoJSON
gdf.to_file("lines_retreat_1.9_2005_50percentile.geojson", driver="GeoJSON")
