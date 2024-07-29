import geopandas as gpd
from shapely.geometry import LineString
import math
from shapely.ops import linemerge
from shapely.geometry import LineString, Point,MultiLineString
def multiline_to_linestring(geometry):
    if isinstance(geometry, MultiLineString):
        # Attempt to merge the lines into a single LineString
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return merged
        else:
            # If merge is not possible, return the original or handle differently
            return geometry
    return geometry

def line_to_segments(line):
    return [LineString([line.coords[i], line.coords[i + 1]]) for i in range(len(line.coords) - 1)]
def calculate_bearing(segment):
    if isinstance(segment, LineString):
        start, end = segment.coords[0], segment.coords[-1]
        angle = math.atan2(end[0] - start[0], end[1] - start[1])
        bearing = (math.degrees(angle) + 360) % 360
        return bearing
    return None
def point_line_position(point, line):
    """Returns 'left', 'right', or 'on' based on the position of the point relative to the line."""
    line_start, line_end = list(line.coords)
    vector_line = [line_end[0] - line_start[0], line_end[1] - line_start[1]]
    vector_point = [point.x - line_start[0], point.y - line_start[1]]
    cross_product = vector_line[0] * vector_point[1] - vector_line[1] * vector_point[0]
    
    if cross_product > 0:
        return 'left'
    elif cross_product < 0:
        return 'right'
    else:
        return 'on'
    
def calculate_orientation(row):
    if row['orientation'] == 'right':
        return (row['bearing_l'] - 90)% 360
    elif row['orientation'] == 'left':
        return (row['bearing_l'] +90 )% 360