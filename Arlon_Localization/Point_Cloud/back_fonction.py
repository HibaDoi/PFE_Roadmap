from utils_Final import *
import numpy as np

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