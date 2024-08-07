import laspy
import os
from PIL import Image
import numpy as np
import pandas as pd 
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.spatial import distance
from back_fonction import reprojection
from ultralytics import YOLO

las = laspy.read("C:/PFE_Roadmap/point-cloud/o.las")

# Extract Z values and segmentID
z_values = las.z
x_values = las.x
y_values = las.y

segment_ids = las.sid
print("Scale:", las.header.scale)
print("Offset:", las.header.offset)
######################################################################"
unique_segment_ids = np.unique(segment_ids)
input(unique_segment_ids)
for segment_id in unique_segment_ids:
    segment_mask = segment_ids == segment_id
    segment_z_values = z_values[segment_mask]
    segment_x_values = x_values[segment_mask]
    segment_y_values = y_values[segment_mask]
    height_range = np.max(segment_z_values) - np.min(segment_z_values)
    print(height_range)