from utils_localisation_Arlon import *
import json
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from localization import XYlocation
img_width=7040
img_height=3520
labels="C:/visulisation_detection_final/lables.txt"
camera_info_file="Roll_pitch_Yaw_Arlon.csv"
# Open the file in read mode
with open(labels, 'r') as file:
    # Read lines into a list, stripping the newline character
    label = [line.strip() for line in file]
print(label)



for i in range(len(label)):
    dossier=os.path.join("C:/visulisation_detection_final/ALL_Classes_of_Traffic_sign_Arlon",str(label[i]))
    dossier_json=os.path.join(dossier,str(label[i])+str("_json"))
    directory_path = dossier_json
    file_path = os.path.join('Arlon_Localization/geoparquet',"1_raw_points_"+str(label[i])+".csv")
    XYlocation(camera_info_file,directory_path,file_path)
    













    