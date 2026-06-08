
""" Creates polygons from prev generated percentile bands"""
#%%
import geopandas as gpd
from shapely.geometry import LineString, Polygon


proj_year =  2100

scenarios = [1.9,4.5,8.5]
shore_scens = ['oslr','htrend']
for scenario in scenarios:
    for shore_scen in shore_scens:
        # Load both line GeoJSONs
        gdf1 = gpd.read_file(f"../{shore_scen}_lines_shoreline_17_shiftedqtl_{proj_year}_{scenario}.geojson")
        gdf2 = gpd.read_file(f"../{shore_scen}_lines_shoreline_83_shiftedqtl_{proj_year}_{scenario}.geojson")

        polygons = []

        for group_idx, row in gdf1.iterrows():
            line_group_id = row["group_id"]
            line1 = row.geometry
            line2 = gdf2.loc[gdf2["group_id"] == line_group_id, "geometry"].values

            if len(line2) == 0:
                print(f"⚠️ No match for group_id {line_group_id}")
                continue

            line2 = line2[0]

            # Create polygon between both lines
            coords = list(line1.coords) + list(line2.coords)[::-1]
            poly = Polygon(coords)
            polygons.append({"group_id": line_group_id, "geometry": poly})

        poly_gdf = gpd.GeoDataFrame(polygons, crs=gdf1.crs)
        poly_gdf.to_file(f"polygons_{shore_scen}_between_percentiles_{proj_year}_{scenario}.geojson", driver="GeoJSON")
# %%
