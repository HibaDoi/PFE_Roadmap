import geopandas as gpd
from shapely.geometry import LineString,Point, LineString
from shapely.ops import nearest_points
import pandas as pd
from utils import *
# Charger le fichier SHP
points_gdf = gpd.read_file("points/Lamppost_single.shp")
points_gdf ['FID']=range(len(points_gdf))
segments_line_gdf= gpd.read_parquet('road_segments_with_bearing.geoparquet')
pd.set_option('display.max_colwidth', None)

# Create a spatial index for the lines layer to improve performance
sindex = segments_line_gdf.sindex
# Check and add necessary fields
if 'proj_x' not in points_gdf.columns:
    points_gdf['proj_x'] = pd.NA
if 'proj_y' not in points_gdf.columns:
    points_gdf['proj_y'] = pd.NA
if 'id_line' not in points_gdf.columns:
    points_gdf['id_line'] = pd.NA
if 'bearing_l' not in points_gdf.columns:
    points_gdf['bearing_l'] = pd.NA
# Process each point
for index, point in points_gdf.iterrows():
    point_geom = point.geometry
    # Using the spatial index to narrow down the search area
    possible_matches_index = list(sindex.nearest(point_geom, max_distance=30, return_distance=False)  )
    possible_matches = segments_line_gdf.loc[possible_matches_index[1]]
    min_dist = float('inf')
    snapped_point = None
    # Iterate over possible line segments
    for _, line in possible_matches.iterrows():
        line_geom = line.geometry
        # Check each segment of the line
        proj_point = nearest_points(point_geom, line_geom)[1]
        dist = point_geom.distance(proj_point)
        if dist < min_dist:
            min_dist = dist
            snapped_point = proj_point
            idd=line.id  
            gisement=line.bearing
    # Update the GeoDataFrame
    if snapped_point:
        points_gdf.at[index, 'proj_x'] = snapped_point.x
        points_gdf.at[index, 'proj_y'] = snapped_point.y
        points_gdf.at[index, 'distance'] = min_dist
        points_gdf.at[index, 'id_line'] = idd
        points_gdf.at[index, 'bearing_l'] = gisement

DD=[]
ii=[]
for i  in range(len(points_gdf)):
    t=point_line_position(points_gdf.iloc[i]["geometry"], segments_line_gdf.iloc[points_gdf.iloc[i]['id_line']]["geometry"])
    DD.append(t)
points_gdf['orientation']=DD
points_gdf['G_obj'] = points_gdf.apply(calculate_orientation, axis=1)
points_gdf.to_parquet('point_with_orientation.geoparquet')

