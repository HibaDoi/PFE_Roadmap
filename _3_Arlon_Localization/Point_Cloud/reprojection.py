import laspy
import os
from PIL import Image
import numpy as np
import pandas as pd 
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from scipy.spatial import distance
from back_fonction import *
from ultralytics import YOLO
from scipy.spatial.distance import euclidean
dir_img_arlon="C:/Users/Administrateur/pfe_hiba_workspace/PFE_HIBA/Image_Arlon"
model = YOLO("C:/Users/Administrateur/pfe_hiba_workspace/runs/classify/FOR_point_cloud/weights/best.pt")
IMG_W=7040
IMG_H=3520
las = laspy.read("point-cloud/offset/final_segment_Hc_correct.las")
print("Scale:", las.header.scale)
print("Offset:", las.header.offset)
# Read the CSV file
csv_file_path = 'Roll_pitch_Yaw_Arlon.csv'  # Replace with your actual CSV file path
dff = pd.read_csv(csv_file_path)
geometry = [Point(xy) for xy in zip(dff['x'], dff['y'])]
gdf = gpd.GeoDataFrame(dff, geometry=geometry)
# Optionally, set a coordinate reference system (CRS) if you know it
gdf.set_crs(epsg=31370, inplace=True)  

# Extract Z values and segmentID
z_values = las.z
x_values = las.x
y_values = las.y
segment_ids = las.sid
unique_segment_ids = np.unique(segment_ids)
segment = pd.DataFrame()
id=[]
height=[]
X=[]
Y=[]
Z=[]
for segment_id in unique_segment_ids:
    segment_mask = segment_ids == segment_id
    segment_z_values = z_values[segment_mask]
    segment_x_values = x_values[segment_mask]
    segment_y_values = y_values[segment_mask]
    height_range = np.max(segment_z_values) - np.min(segment_z_values)
    
    if height_range >4:
        id.append(segment_id)
        height.append(height_range)
        print(height_range)
        xyz = np.vstack((segment_x_values,segment_y_values,segment_z_values)).transpose()
        df = pd.DataFrame(xyz, columns=['X', 'Y', 'Z'])
        min_z_point = df.loc[df['Z'].idxmin()]
        gdf['distance'] = gdf.apply(lambda row: euclidean((min_z_point['X'], min_z_point['Y']), (row['x'], row['y'])), axis=1)
        closest_points = gdf.nsmallest(4, 'distance')
        X.append(min_z_point['X'])
        Y.append(min_z_point['Y'])
        Z.append(min_z_point['Z'])
        
        bounding_box_vertices = get_bounding_box_vertices(las,segment_x_values,segment_y_values,segment_z_values)
        
        for i in range(4):
            pixel_coord=[]
            camera_orienration=[closest_points.iloc[i]['roll'],closest_points.iloc[i]['Pitch'],closest_points.iloc[i]['Yaw']]
            XYZ=[closest_points.iloc[i]['x'],closest_points.iloc[i]['y'],closest_points.iloc[i]['z']]

            for coor in bounding_box_vertices:
                wl=reprojection(coor,camera_orienration,XYZ,IMG_W,IMG_H)
                pixel_coord.append(wl)
            points=np.array(pixel_coord)
            # Calculate the bounding box with padding
            padding = 20  # Adjust this value to add more or less space

            min_x = np.min(points[:, 0]) - padding
            max_x = np.max(points[:, 0]) + padding
            min_y = np.min(points[:, 1]) - padding
            max_y = np.max(points[:, 1]) + padding
            # Calculate center, width, and height
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            bbox_width = max_x - min_x
            bbox_height = max_y - min_y
            # Normalize the values
            norm_center_x = center_x / IMG_W
            norm_center_y = center_y / IMG_H
            norm_bbox_width = bbox_width / IMG_W
            norm_bbox_height = bbox_height / IMG_H
            # Format the YOLO line (assuming class ID 0)
            yolo_format = f"0 {norm_center_x} {norm_center_y} {norm_bbox_width} {norm_bbox_height}"
            # Write to a .txt file
            with open("a.txt", "w") as file:
                file.write(yolo_format)
            img_arlon=os.path.join(dir_img_arlon,closest_points.iloc[i]['nom'][:-4]+".jpg")
segment["segment_id"]=id
segment["height"]=height
segment["X"]=X
segment["Y"]=Y
segment["Z"]=Z
print(segment)

gdf = gpd.GeoDataFrame(segment, geometry=gpd.points_from_xy(segment['X'], segment['Y']))

# Set the CRS (for example, WGS 84)
gdf = gdf.set_crs("EPSG:31370")

# Save the GeoDataFrame as a GeoParquet file
gdf.to_parquet('Arlon_Localization/Point_Cloud/instance_xyzh_lamppost.geoparquet')
            # crop_images_based_on_yolo(img_arlon,"a.txt", f"Arlon_Localization/Point_Cloud/cropped_image/obj{segment_id}__{i}")
        
        # #####################################################################
        # results = model.predict(source='output_path_base_ccccc_1.jpg',save_txt=True, conf=0.25)
        # input("sir chof cropped")
        # with open(os.path.join(results[0].save_dir,os.path.join("labels",'obj.txt')), 'r') as file:
        #     line = file.readline()   
        #     print(line)
        #     #input("??????????????????????????????????")
        #     stat ,clas =  line.split()
        #     print(clas)

        # if os.path.exists(os.path.join(results[0].save_dir,os.path.join("labels",'output_path_base_ccccc_1.txt'))):
        # # Delete the file
        #     os.remove(os.path.join(results[0].save_dir,os.path.join("labels",'output_path_base_ccccc_1.txt')))
        # import geopandas as gpd
        # from shapely.geometry import Point

        # # Define an empty GeoDataFrame with specified columns
        # columns = ['Name', 'x', 'y','z', 'geometry']
        # gdf = gpd.GeoDataFrame(columns=columns)
        # gdf.set_geometry('geometry', inplace=True)
        # gdf.set_crs(epsg=31370, inplace=True)  # Set CRS to WGS84
        # # for name, lat, lon in locations:
        #     # Create a Point geometry
        # point = Point(closest_points.iloc[0]['x'],closest_points.iloc[0]['y'])

        # # Create a DataFrame for the new entry
        # new_data = gpd.GeoDataFrame([[clas,closest_points.iloc[0]['x'],closest_points.iloc[0]['y'],closest_points.iloc[0]['z'], point]],
        #                             columns=columns,
        #                             crs=gdf.crs)  # Ensure new data has the same CRS

        # # Append the new data to the GeoDataFrame


        # print(new_data)
        # input()