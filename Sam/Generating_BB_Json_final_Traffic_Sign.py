import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
from utils_Generating_Mask_and_Json_final import *
from segment_anything import sam_model_registry, SamPredictor
import re
from PIL import Image
import numpy as np
import json
img_width=7040
img_height=3520
labels="C:/visulisation_detection_final/lables.txt"
# Open the file in read mode
with open(labels, 'r') as file:
    # Read lines into a list, stripping the newline character
    label = [line.strip() for line in file]
print(label)
target_directory = "C:/visulisation_detection_final/ALL_Classes_of_Traffic_sign_Arlon"
for i in range(len(label)):
    dossier=os.path.join(target_directory,str(label[i]))
    print(dossier)
    dossier_json=os.path.join(dossier,str(label[i])+str("_json"))
    print(dossier_json)


    def bounding_box_centre_json(dossier,dossier_json,img_width,img_height):
        print("aaaaaaaaaaaaaa")
        if not os.path.exists(dossier_json):
                os.makedirs(dossier_json)
        #categorie lammpost
        k=1
        #############################################################
        # Liste tous les fichiers dans le dossier spécifié
        fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
        fichiers.sort(key=natural_sort_key)
        p=-1

        def pixel_coordinat_bb(f,w,h):
            return f[0]*w,f[1]*h
        # Boucle à travers chaque fichier
        for fichier in fichiers:
            chemin_fichier = os.path.join(dossier, fichier)
            with open(chemin_fichier, 'r') as file:  
                root, _ = os.path.splitext(chemin_fichier)
                # Add the new extension
                lignes = file.readlines()
                lignes = [line for line in lignes if line.startswith('0')]
                l=[]
                if lignes != []:
                    for i in range(len(lignes)):
                        data_str = lignes[i].strip()
                        # Split the string by spaces
                        data_list = data_str.split()
                        # Convert the list of strings to a list of floats
                        data_floats = [float(x) for x in data_list]
                        # Remove the first element as it's not required
                        result = data_floats[1:]
                        print(result)
                        #input("//////////////////////////////////////:")

                        gg=pixel_coordinat_bb(result,img_width,img_height)
                        print(gg)
                        #input("/////////////////////////////////////////")
                        l.append(gg)
                    print(l)
                    #input("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")


                    results = []  # List to store results
                    mask_id = 1 
                    for L in l:
                        ##########################################################

                        results.append({
                            "id": mask_id,
                            "isthing": True,
                            "category_id": k,
                            "xy": L
                        })
                        # Increment mask ID for the next mask
                        mask_id += 1
                        print(f'Bottom pixel coordinates: {L}')
                    chemin_fichier = chemin_fichier.replace("\\", "/")
                    path_parts1 = chemin_fichier.split('/')
                    filename_json = os.path.splitext(path_parts1[-1])[0] + ".json"  # Change the file extension from .jpg to .png
                    output_path1=os.path.join(dossier_json,filename_json)
                    # Writing the results to a JSON file
                    with open(output_path1, 'w') as f:
                        json.dump(results, f)
                    
                    print("JSON file has been written with mask details.")
                        
    bounding_box_centre_json(dossier,dossier_json,img_width,img_height)
                    