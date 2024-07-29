import numpy as np
import cv2
from math import *
from pyproj import Transformer,CRS
def resize_image(image, max_width=1600, max_height=1300):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image, scaling_factor
    return image, 1.0
def select_points(image, window_name):
    points = []
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, click_event)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return points

import numpy as np
import math
def degree_to_rad0(deg):
    return deg * np.pi / 200
def degree_to_rad(deg):
    return deg * np.pi / 180
def rad_to_degree(rad):
    return rad / math.pi * 180
    
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
    if real_heading_degree < 0:
        real_heading_degree += 360
    return real_heading_degree
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
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)