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

#############################################################
dossier = "C:/visulisation_detection_final/Arlon_Lamppost"
dossier_json="C:/visulisation_detection_final/Arlon_Lamppost/json_single_adding_hauteur"
dossier_mask="C:/visulisation_detection_final/Arlon_Lamppost/mask_single_adding_hauteur"
#categorie single
line_start='0'
#categorie lammpost
k=1
img_width=7040
img_height=3520
#############################################################
sam_checkpoint = "C:/Users/Administrateur/pfe_hiba_workspace/SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)
# Liste tous les fichiers dans le dossier spécifié
fichiers = [f for f in os.listdir(dossier) if f.endswith('.txt')]
fichiers.sort(key=natural_sort_key)
p=-1
num_colors = 10  # Example number of colors, you can change as needed
palette = []
for i in range(num_colors):
    # Generate a random RGB color tuple for each index
    color = tuple(np.random.randint(0, 256, size=3))
    palette.extend(color)
# Boucle à travers chaque fichier
for fichier in fichiers:
    chemin_fichier = os.path.join(dossier, fichier)
    with open(chemin_fichier, 'r') as file:  
        root, _ = os.path.splitext(chemin_fichier)
        # Add the new extension

        new_path = root + ".jpg"

        lignes = file.readlines()
        lignes = [line for line in lignes if line.startswith(line_start)]
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

                gg=bbyolo2xyxy(result,img_width,img_height)
                l.append(gg)
            image = preprocess_image(new_path)
            predictor.set_image(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_boxes = torch.tensor(l, device=predictor.device)
            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            results = []  # List to store results
            mask_id = 1 
            for mask in masks:
                ##########################################################
                image_array = mask.cpu().numpy()

                # Find the coordinates of the bottom pixel in the mask
                # The mask is represented by pixels with a value of 1
                mask = image_array[0] == True
                # Get the bottom-most pixel's coordinates
                indices = np.where(mask)
                if indices[0].size > 0:
                    y_bottom = indices[0].max()
                    m=indices[0]==y_bottom
                    mm = np.where(m)
                    median_value = np.median(mm)
                    bottom_pixel = (int(indices[1][int(median_value)]), int(y_bottom))
                    top_pixel=(int(indices[1][0]),int(indices[0][0]))
                else:
                    bottom_pixel = None
                    # Append the result for this mask to the results list
                results.append({
                    "id": mask_id,
                    "isthing": True,
                    "category_id": k,
                    "xy": bottom_pixel,
                    "xyt":top_pixel
                })
                # Increment mask ID for the next mask
                mask_id += 1
                print(f'Bottom pixel coordinates: {bottom_pixel}')
            chemin_fichier = chemin_fichier.replace("\\", "/")
            path_parts1 = chemin_fichier.split('/')
            filename_json = os.path.splitext(path_parts1[-1])[0] + ".json"  # Change the file extension from .jpg to .png
            output_path1=os.path.join(dossier_json,filename_json)
            # Writing the results to a JSON file
            with open(output_path1, 'w') as f:
                json.dump(results, f)
            
            print("JSON file has been written with mask details.")
                

            ######################################################## 
            data = np.zeros((img_height,img_width), dtype=np.uint8)
            masks =masks.cpu()
            masks = masks.numpy()
            for i in range(len(masks)):
                data[masks[i][0] > 0] = (i+1)
            # Define the number of colors you want
            
            # Create an 8-bit color image from the NumPy array
            image = Image.fromarray(data, mode='P')
            # Define a color palette for each color index

            # Fill the remaining palette entries with zeros
            palette += [0] * (256 * 3 - len(palette))
            # Apply the palette to the image
            image.putpalette(palette)
            ##################
            chemin_fichier = chemin_fichier.replace("\\", "/")
            path_parts = chemin_fichier.split('/')
            # Change the directory and file extension
            
            filename_png = os.path.splitext(path_parts[-1])[0] + ".png"  # Change the file extension from .jpg to .png
            # Rejoin the path
            output_path = output_path1=os.path.join(dossier_mask,filename_png)
            ##################
            # Save the image as a PNG file
            image.save(output_path)