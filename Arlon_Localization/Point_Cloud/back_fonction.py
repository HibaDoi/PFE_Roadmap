from utils_Final import *
import numpy as np
from PIL import Image
def reprojection(point,orien_coor,XYZ_coor,img_width,img_height):
    #coordonnées de camera dans le repere rectangulaire
    x,y,z=XYZ_coor
    
    Orientation_degree=[0,0,0]
    Orientation_degree[0]=orien_coor[0]
    Orientation_degree[1]=orien_coor[1]
    Orientation_degree[2]=-orien_coor[2]
    A=np.array(get_rotation_matrix_by_euler_angles(Orientation_degree[0],Orientation_degree[1],Orientation_degree[2]))
    A_inv = np.linalg.inv(A)
    xd,yd,zd=point[0]-x,point[1]-y,point[2]-z
    X=np.array([[yd],[xd],[zd]])
    nn=np.dot(A_inv,X)
    # Calculate r (radius)
    r1 = np.sqrt(nn[0]**2 + nn[1]**2 + nn[2]**2)
    nn=1/r1*nn
    r=1
    # Calculate L_rad (azimuthal angle), using atan2 for correct quadrant handling
    B_rad = np.arctan2(nn[1], nn[0])[0]
  
    # Calculate B_rad (polar angle)
    L_rad = -np.arcsin(nn[2] / r)[0]  # Angle from the positive z-axis
    # Or alternatively, if you need the elevation from the xy-plane:
    # B_rad = np.arctan2(np.sqrt(x**2 + y**2), z)
    # Output the spherical coordinates
    # Image dimensions and reference point coordinates
    
    w0 = img_width / 2
    h0 = img_height / 2
    # Calculate the pixel coordinates W and l
    W = (B_rad * img_width /(2*pi))+w0 
    l = ((L_rad * img_height /pi))+h0
    return [W,l]


def crop_images_based_on_yolo(image_path, annotation_path, output_path_base):
    img = Image.open(image_path)
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
        
    # Parse the YOLO format data: class x_center y_center width height (normalized)
    _, x_center, y_center, width, height = map(float, lines[0].split())
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


def get_bounding_box_vertices(laz_file_path,segment_x_values,segment_y_values,segment_z_values):
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