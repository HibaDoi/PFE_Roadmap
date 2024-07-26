import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from utils_Generating_Mask_and_Json_final import *

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Resize image to standard equirectangular dimensions
    # Denoise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # Enhance contrast using CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  
def bbyolo2xyxy(bb):
    # Image dimensions
    image_width, image_height = 8192,4096
    
    # Convert YOLO format to xyxy format
    center_x = bb[0] * image_width
    center_y = bb[1] * image_height
    width = bb[2] * image_width
    height = bb[3] * image_height
    
    x_min = center_x - width / 2
    y_min = center_y - height / 2
    x_max = center_x + width / 2
    y_max = center_y + height / 2
    
    bbox_xyxy = [x_min, y_min, x_max, y_max]
    return bbox_xyxy

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]