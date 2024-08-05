import geopandas as gpd
from shapely.geometry import LineString,Point, LineString
from shapely.ops import nearest_points
import pandas as pd
from utils import *
# Charger le fichier SHP
line_gdf = gpd.read_file("Features_extractions_orientation/roads/road_liege_31370_sans_nett_sans_expolose.shp")
line_gdf = line_gdf[~line_gdf['highway'].isin(['pedestrian', 'path','footway','cycleway',"steps"])]
line_gdf=line_gdf.explode()
segments_list = []
i=0
for index, row in line_gdf.iterrows():
    segments = multiline_to_linestring(row.geometry)
    Segments = line_to_segments(segments)
    for Segment in Segments:
        segments_list.append({'geometry': Segment, 'id': i})
        i+=1
segments_line_gdf = gpd.GeoDataFrame(segments_list, geometry='geometry')
segments_line_gdf.set_crs(epsg=31370, inplace=True)
segments_line_gdf['bearing'] = segments_line_gdf['geometry'].apply(calculate_bearing)
# Save to a Shapefile
segments_line_gdf.to_parquet("road_segments_with_bearing.geoparquet")
