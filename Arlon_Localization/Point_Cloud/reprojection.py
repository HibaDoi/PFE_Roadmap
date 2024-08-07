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

model = YOLO("C:/Users/Administrateur/pfe_hiba_workspace/runs/classify/FOR_point_cloud/weights/best.pt")
IMG_W=7040
IMG_H=3520
las = laspy.read("C:/Users/Administrateur/Desktop/KPConv-PyTorch/segmentedmerged/outputmerged/final_track11.las")
# Read the CSV file
csv_file_path = 'Roll_pitch_Yaw_Arlon.csv'  # Replace with your actual CSV file path
#######################################################################################
def crop_images_based_on_yolo(image_path, annotation_path, output_path_base):
    img = Image.open(image_path)
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        input(lines)
    for i, line in enumerate(lines):
        # Parse the YOLO format data: class x_center y_center width height (normalized)
        _, x_center, y_center, width, height = map(float, line.split())

        # Get dimensions of the original image
        img_w, img_h = img.size

        # Convert YOLO coordinates to pixel coordinates
        box_w = width * img_w
        box_h = height * img_h
        box_x_center = x_center * img_w
        box_y_center = y_center * img_h

        # Define the bounding box (left, upper, right, lower)
        left = int(box_x_center - (box_w / 2))
        upper = int(box_y_center - (box_h / 2))
        right = int(box_x_center + (box_w / 2))
        lower = int(box_y_center + (box_h / 2))
        box = (left, upper, right, lower)

        # Crop the image
        cropped_img = img.crop(box)

        # Save the cropped image
        cropped_img.save(f"{output_path_base}.jpg")
#######################################################################################
# Extract Z values and segmentID
z_values = las.z
x_values = las.x
y_values = las.y

segment_ids = las.segmentID
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

    ######################################################################"
    if height_range>1:
    # Extract XYZ values
        xyz = np.vstack((segment_x_values,segment_y_values,segment_z_values)).transpose()

        # Create a DataFrame for easier manipulation
        df = pd.DataFrame(xyz, columns=['X', 'Y', 'Z'])

        # Group by X and Y, then find the index of the minimum Z value in each group
        min_z_point = df.loc[df['Z'].idxmin()]

        # Display the resulting bottom points
        print(min_z_point)




        dff = pd.read_csv(csv_file_path)

        # Assuming your CSV has columns 'X', 'Y', and 'Z' for coordinates
        geometry = [Point(xy) for xy in zip(dff['x'], dff['y'])]
        gdf = gpd.GeoDataFrame(dff, geometry=geometry)

        # Optionally, set a coordinate reference system (CRS) if you know it
        gdf.set_crs(epsg=31370, inplace=True)  # WGS 84


        ##############################################################################
        # Calculate the Euclidean distance between the given object and all points in the GeoDataFrame
        gdf['distance'] = gdf.apply(lambda row: distance.euclidean((min_z_point['X'], min_z_point['Y']), (row['x'], row['y'])), axis=1)

        # Find the four closest points
        closest_points = gdf.nsmallest(4, 'distance')

        # Display the closest points
        print(closest_points)
        ##############################################################################

        def get_bounding_box_vertices(laz_file_path):
            # Open the LAZ file

            min_x, min_y, min_z = segment_x_values.min(), segment_y_values.min(), segment_z_values.min()
            max_x, max_y, max_z = segment_x_values.max(), segment_y_values.max(), segment_z_values.max()

            # Calculate all eight vertices of the bounding box
            vertices = [
                (min_x, min_y, min_z),
                (min_x, min_y, max_z),
                (min_x, max_y, min_z),
                (min_x, max_y, max_z),
                (max_x, min_y, min_z),
                (max_x, min_y, max_z),
                (max_x, max_y, min_z),
                (max_x, max_y, max_z),
            ]
            return vertices

        # Get the coordinates of all eight vertices of the bounding box
        bounding_box_vertices = get_bounding_box_vertices(las)
        pixel_coord=[]
        camera_orienration=[closest_points.iloc[0]['roll'],closest_points.iloc[0]['Pitch'],closest_points.iloc[0]['Yaw']]
        XYZ=[closest_points.iloc[0]['x'],closest_points.iloc[0]['y'],closest_points.iloc[0]['z']]
        for coor in bounding_box_vertices:
            wl=reprojection(coor,camera_orienration,XYZ,IMG_W,IMG_H)
            pixel_coord.append(wl)

        points=np.array(pixel_coord)
        # Calculate min and max for x and y
        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

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
        with open(closest_points.iloc[0]['nom'][:-4]+".txt", "w") as file:
            file.write(yolo_format)
        dir_img_arlon="C:/Users/Administrateur/pfe_hiba_workspace/PFE_HIBA/Image_Arlon"
        img_arlon=os.path.join(dir_img_arlon,closest_points.iloc[0]['nom'][:-4]+".jpg")
        crop_images_based_on_yolo(img_arlon,closest_points.iloc[0]['nom'][:-4]+".txt", f"obj{segment_id}")

        results = model.predict(source='output_path_base_ccccc_1.jpg',save_txt=True, conf=0.25)
        input("sir chof cropped")
        with open(os.path.join(results[0].save_dir,os.path.join("labels",'obj.txt')), 'r') as file:
            line = file.readline()   
            print(line)
            #input("??????????????????????????????????")
            stat ,clas =  line.split()
            print(clas)

        if os.path.exists(os.path.join(results[0].save_dir,os.path.join("labels",'output_path_base_ccccc_1.txt'))):
        # Delete the file
            os.remove(os.path.join(results[0].save_dir,os.path.join("labels",'output_path_base_ccccc_1.txt')))
        import geopandas as gpd
        from shapely.geometry import Point

        # Define an empty GeoDataFrame with specified columns
        columns = ['Name', 'x', 'y','z', 'geometry']
        gdf = gpd.GeoDataFrame(columns=columns)
        gdf.set_geometry('geometry', inplace=True)
        gdf.set_crs(epsg=31370, inplace=True)  # Set CRS to WGS84
        # for name, lat, lon in locations:
            # Create a Point geometry
        point = Point(closest_points.iloc[0]['x'],closest_points.iloc[0]['y'])

        # Create a DataFrame for the new entry
        new_data = gpd.GeoDataFrame([[clas,closest_points.iloc[0]['x'],closest_points.iloc[0]['y'],closest_points.iloc[0]['z'], point]],
                                    columns=columns,
                                    crs=gdf.crs)  # Ensure new data has the same CRS

        # Append the new data to the GeoDataFrame


        print(new_data)
        input()