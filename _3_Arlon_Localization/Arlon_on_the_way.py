import numpy as np
import cv2
from math import *
from pyproj import Transformer,CRS
from Arlon_utils_ontheway import *
# Load images
image9 = cv2.imread('F:/laz-20240711T122904Z-001/laz/images/Job_20240422_0938_Track11_Sphere_00061.jpg')
image10 = cv2.imread('F:/laz-20240711T122904Z-001/laz/images/Job_20240422_0938_Track11_Sphere_00062.jpg')
image11 = cv2.imread('F:/laz-20240711T122904Z-001/laz/images/Job_20240422_0938_Track11_Sphere_00063.jpg')

# Resize images if necessary
resized_image1, scale_factor1 = resize_image(image9)
resized_image2, scale_factor2 = resize_image(image10)
resized_image3, scale_factor3 = resize_image(image11)

# Select points in resized images
points_image1 = select_points(resized_image1.copy(), 'Select points in image 9')
points_image2 = select_points(resized_image2.copy(), 'Select points in image 10')
points_image3 = select_points(resized_image3.copy(), 'Select points in image 11')
# Adjust points to original image size
points_image1 = [(int(x / scale_factor1), int(y / scale_factor1)) for x, y in points_image1]
points_image2 = [(int(x / scale_factor2), int(y / scale_factor2)) for x, y in points_image2]
points_image3 = [(int(x / scale_factor3), int(y / scale_factor3)) for x, y in points_image3]
p1=points_image1[0]
p2=points_image2[0]
p3=points_image3[0]
P=[p1,p2,p3]
Camera_Orientation=[
[-3.1961844057013944,1.829367593225741,-106.3633296899049],
[-2.9858203456570256,1.881200447433476,-105.55433064100468],
[-2.494331000718142,2.569904909952329,-103.53678166461411]
]

gisement=[]
for i in range(3):
    
    o1=calculate_heading(P[i][0],P[i][1],7040,3520, -Camera_Orientation[i][0],-Camera_Orientation[i][1],-Camera_Orientation[i][2])*200/180
    print(o1)
    gisement.append(o1)
#k =[longitude,latitude,height,roll,pitch,yaw] for image 296.297.298


Camera_Coordinate_Lambert_72=[
[253636.4071944703,41955.5752021298,392.0993416049],
[253639.2937488526,41954.6234610658,392.10052751],
[253642.2058329791,41953.7046426404,392.0793069284]
]
def droit(point,l):
    a=tan(l)
    b=point[1]-a*point[0]
    return a,b
def intersection(a1, b1, a2, b2):
    # Vérifier si les droites sont parallèles
    if a1 == a2:
        return None  # Les droites sont parallèles et n'ont pas d'intersection

    # Calculer x
    x = (b2 - b1) / (a1 - a2)
    
    # Calculer y
    y = a1 * x + b1
    
    return (x, y)

Par_droit=[]
for i in range(3):
    Par_droit.append(droit(Camera_Coordinate_Lambert_72[i],(100-gisement[i])*pi/200))
C1=intersection(Par_droit[0][0], Par_droit[0][1], Par_droit[1][0], Par_droit[1][1])
print(intersection(Par_droit[0][0], Par_droit[0][1], Par_droit[1][0], Par_droit[1][1]))
C2=intersection(Par_droit[0][0], Par_droit[0][1], Par_droit[2][0], Par_droit[2][1])
print(intersection(Par_droit[0][0], Par_droit[0][1], Par_droit[2][0], Par_droit[2][1]))
C3=intersection(Par_droit[1][0], Par_droit[1][1], Par_droit[2][0], Par_droit[2][1])
print(intersection(Par_droit[1][0], Par_droit[1][1], Par_droit[2][0], Par_droit[2][1]))


print(distance(C1,C2))
print(distance(C1,C3))
print(distance(C2,C3))