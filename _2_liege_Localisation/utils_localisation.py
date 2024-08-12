import numpy as np
from math import *
from pyproj import Transformer,CRS
import re
from pathlib import Path
from collections import deque
import re
import numpy as np
import math
from sklearn.cluster import DBSCAN
import pandas as pd
def degree_to_rad(deg):
    return deg * np.pi / 180
########################################################
def rad_to_degree(rad):
    return rad / math.pi * 180
########################################################
def calculate_heading(detect_obj_x, detect_obj_y, img_width, img_height, roll, pitch, heading):
    # Calculate center points of the image
    w0, h0 = img_width / 2, img_height / 2

    # Calculate angle in degrees relative to the center of the image
    B_degree = (detect_obj_x - w0) * (360.0 / img_width)
    L_degree = (detect_obj_y - h0) * (180.0 / img_height)

    # Convert degrees to radians
    B_rad = degree_to_rad(B_degree)
    L_rad = degree_to_rad(L_degree)
    # Spherical projection (assuming sphere radius of 1 for simplicity)
    r = 1
    t = r * np.cos(L_rad)
    x = t * np.cos(B_rad)
    y = t * np.sin(B_rad)
    z = r * np.sin(L_rad)

    # Assuming xyz are coordinates in the camera's coordinate system
    xyz = np.array([[x], [y], [z]])
    R = get_rotation_matrix_by_euler_angles(roll, pitch, heading)
    world_coord = np.dot(R, xyz)
    real_heading_rad = math.atan2(world_coord[1, 0], world_coord[0, 0])
    real_heading_degree = rad_to_degree(real_heading_rad)
    angle_vertical = math.atan(world_coord[2, 0]/math.sqrt( world_coord[0, 0]**2+world_coord[1, 0]**2))
    real_heading_degree = rad_to_degree(real_heading_rad)
    angle_vertical = rad_to_degree(angle_vertical)
    return [real_heading_degree%360,angle_vertical%360]
########################################################
def get_rotation_matrix_by_euler_angles(roll_degree, pitch_degree, heading_degree):
    roll_rad = degree_to_rad(roll_degree)
    pitch_rad = degree_to_rad(pitch_degree)
    heading_rad = degree_to_rad(heading_degree)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll_rad), -math.sin(roll_rad)],
        [0, math.sin(roll_rad), math.cos(roll_rad)]
    ])

    Ry = np.array([
        [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
        [0, 1, 0],
        [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
    ])

    Rz = np.array([
        [math.cos(heading_rad), -math.sin(heading_rad), 0],
        [math.sin(heading_rad), math.cos(heading_rad), 0],
        [0, 0, 1]
    ])

    RotationMatrix = np.dot(np.dot(Rz, Ry), Rx)
    return RotationMatrix
#######################################################################
def geographic_to_rectangular(lon, lat, z):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:31370", always_xy=True)
    t = np.array([transformer.transform(lon,lat)[0], transformer.transform(lon,lat)[1], z])
    return t.tolist()
###########################################################################################
def droit(point,l):
    a=tan(l)
    b=point[1]-a*point[0]
    return a,b
###########################################################################################
def intersection(a1, b1, a2, b2):
    # Vérifier si les droites sont parallèles
    if a1 == a2:
        return None  # Les droites sont parallèles et n'ont pas d'intersection

    # Calculer x
    x = (b2 - b1) / (a1 - a2)
    
    # Calculer y
    y = a1 * x + b1
    
    return (x, y)
###########################################################################################
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
############################################################################################
def ObjectCoordinat(Camera_Orientation,Camera_Coordinate_WGS,P):
    gisement=calculate_heading(P[0],P[1],8192, 4096 , Camera_Orientation[0],Camera_Orientation[1],-Camera_Orientation[2]+90)[0]*200/180
    angle_verticale=calculate_heading(P[0],P[1],8192, 4096 , Camera_Orientation[0],Camera_Orientation[1],-Camera_Orientation[2]+90)[1]*200/180
    Camera_Coordinate_Lambert_72=geographic_to_rectangular(Camera_Coordinate_WGS[0],Camera_Coordinate_WGS[1],Camera_Coordinate_WGS[2])
    Par_droit=droit(Camera_Coordinate_Lambert_72,(100-gisement)*pi/200)
    return gisement,angle_verticale,Camera_Coordinate_Lambert_72,Par_droit
####################################################################
def parse_image_info(image_info_file):
    try:
        parsed_data = []
        with open(image_info_file, 'r') as f:
            next(f)  # Skip the header
            for line in f:
                parts = line.strip().split(',')
                parsed_data.append({
                    'imj_lon': float(parts[0].strip()),
                    'imj_lat': float(parts[1].strip()),
                    'imj_height': float(parts[2].strip()),
                    'imj_roll': float(parts[3].strip()),
                    'imj_pitch': float(parts[4].strip()),
                    'imj_yaw': float(parts[5].strip()),
                    'filename': parts[6].strip()
                })
        return parsed_data
    except Exception as e:
        print(f"Error processing file: {e}")
        return []
####################################################################
def numerical_sort_key(s):
    """Extract numerical values as integers to use for sorting."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s.name)]
####################################################################  
def traverse_directory_in_groups(path, group_size=4):
    base_path = Path(path)
    all_files = list(base_path.rglob('*.json'))
    all_files.sort(key=numerical_sort_key)
    return all_files
#######################################
def find_ray_intersection(p1, azimuth1,a1,b1, p2, azimuth2,a2,b2):
    azimuth1=azimuth1*180/200
    azimuth2=azimuth2*180/200
    # Check if the lines are parallel
    if a1 == a2:
        if b1 == b2:
            return "The rays are the same."
        else:
            return "The rays are parallel and do not intersect."
    
    # Calculate the x-coordinate of the intersection of lines
    x = (b2 - b1) / (a1 - a2)
    
    # Calculate the y-coordinate of the intersection
    y = a1 * x + b1
    
    # Check directionality using the dot product condition
    direction1 = (x - p1[0], y - p1[1])
    direction2 = (x - p2[0], y - p2[1])
    reference_direction1 = (cos(radians(azimuth1)), sin(radians(azimuth1)))
    reference_direction2 = (cos(radians(azimuth2)), sin(radians(azimuth2)))

    dot_product1 = direction1[0] * reference_direction1[0] + direction1[1] * reference_direction1[1]
    dot_product2 = direction2[0] * reference_direction2[0] + direction2[1] * reference_direction2[1]

    if dot_product1 > 0 and dot_product2 > 0:
        return (x, y)
    else:
        return None
    
    ##########################################
    #########################################
########################################################
def parse_image_info(image_info_file):
    try:
        parsed_data = []
        with open(image_info_file, 'r') as f:
            next(f)  # Skip the header
            for line in f:
                parts = line.strip().split(',')
                parsed_data.append({
                    'imj_lon': float(parts[0].strip()),
                    'imj_lat': float(parts[1].strip()),
                    'imj_height': float(parts[2].strip()),
                    'imj_roll': float(parts[3].strip()),
                    'imj_pitch': float(parts[4].strip()),
                    'imj_yaw': float(parts[5].strip()),
                    'filename': parts[6].strip()
                })
        return parsed_data
    except Exception as e:
        #print(f"Error processing file: {e}")
        return []
########################################################
def find_intersections(points, rays,ab,V,V2):
    intersectionss = []
    num_points = len(points)

    o=0
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for n in range(len(rays[i])):
                for m in range(len(rays[j])):
                    #print("camera",i,"camera",j,"lob",n,"lob",m)
                    
                    intersection = intersection_of_half_lines(points[i][:2],points[j][:2], ab[i][n], ab[j][m], rays[i][n], rays[j][m])
                    if intersection:
                        o+=1
                        intersectionss.append([intersection, (i, rays[i][n],n),(j, rays[j][m],m),V[i][n],V[j][m],V2[i][n],V2[j][m]])

                            

                        
                    else :
                        print("none")

                    

    
    
    return intersectionss
########################################################
def azimuth_to_angle(azimuth):
    # Convertir l'azimut en angle en radians
    return math.radians(azimuth)
########################################################
def point_from_azimuth(point, azimuth):
    # Obtenir le vecteur directionnel à partir de l'azimut
    angle = azimuth_to_angle(azimuth)
    return (math.cos(angle), math.sin(angle))
########################################################
def intersection_of_lines(ab1, ab2):
    #print(ab1)
    # Calculer le point d'intersection de deux droites (pas des demi-droites)
    a1, b1 = ab1
    a2, b2 = ab2

    return intersection(a1, b1, a2, b2)
########################################################
def intersection_of_half_lines(p1, p2,ab1,ab2,az1,az2):
    # Obtenir les vecteurs directionnels à partir des azimuts
    d1 = ab1[0],ab1[1]
    d2 = ab2[0],ab2[1]

    # Trouver l'intersection des droites étendues
    intersection = intersection_of_lines(ab1,ab2)
    if intersection is None:

        return None

    # Vérifier si l'intersection se trouve dans les deux demi-droites
    n1=appartient_a_la_demi_droite(p1[0], p1[1], intersection[0], intersection[1], az1)
    n2=appartient_a_la_demi_droite(p2[0], p2[1], intersection[0], intersection[1], az2)
    if n1 and n2:
        return intersection

    return None
########################################################
def appartient_a_la_demi_droite(x1, y1, xB, yB, azimuth_grades):
    azimuth_degree=azimuth_grades*180/200
    standard_angle_degrees = 90 - azimuth_degree
    #print(standard_angle_degrees)
    if standard_angle_degrees < 0:
        standard_angle_degrees += 360
    #print(standard_angle_degrees)
    azimuth_radians = (standard_angle_degrees / 360) * 2 * math.pi
    

    direction_x = math.cos(azimuth_radians)
    direction_y = math.sin(azimuth_radians)
   

    # Cas où direction_x est 0
    if direction_x == 0:
        #print("Direction X is zero")
        if (x1 - xB) != 0:
            #print(f"Point A does not align horizontally with point B")
            return False
        t = (y1 - yB) / direction_y
      
        return t >= 0 and direction_y != 0

    # Cas où direction_y est 0
    if direction_y == 0:
        #print("Direction Y is zero")
        if (y1 - yB) != 0:
            #print(f"Point A does not align vertically with point B")
            return False
        t = (x1 - xB) / direction_x
        
        return t >= 0 and direction_x != 0

    # Cas général
    t_x = (x1 - xB) / direction_x
    t_y = (y1 - yB) / direction_y
    
    return abs(t_x - t_y) < 1e-2 and t_x <= 0
#################################################################################

def Total_cluster(input_CSV,output_CSV):
    df = pd.read_csv(input_CSV)
    # Arrondir les colonnes 'x' et 'y' à quatre chiffres après la virgule
    df['centroid_xf'] = df['centroid_xf'].round(5)
    df['centroid_yf'] = df['centroid_yf'].round(5)
    # Supprimer les doublons
    df = df.drop_duplicates(subset=['centroid_xf', 'centroid_yf'])
    # Apply DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=3).fit(df[['centroid_xf', 'centroid_yf']])
    df['cluster'] = clustering.labels_
    # Remove rows where cluster label is -1 (noise)
    df = df[df['cluster'] != -1]
    # Check if the file exists. If it does, append without headers. If not, write with headers.
    try:
        # Attempt to read the file to check if it exists
        pd.read_csv(output_CSV)
        # If it exists, append without writing headers
        df.to_csv(output_CSV, mode='a', header=False, index=False)
    except FileNotFoundError:
        # If the file does not exist, write with headers
        df.to_csv(output_CSV, mode='w', header=True, index=False)
#############################################################################
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_Z(d,v,z):
    return z+d*tan(v*math.pi/200)

def calculate_H(d,v):
    return d*tan(v*math.pi/200)


