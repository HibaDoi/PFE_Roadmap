from utils_localisation import *


# input_CSV='Localisation\csv_file\_1_raw_points_from_localisation_traffic_light.csv'
# output_CSV='Localisation\_2_csv_file\Traffic_sign\_2_unique_points_without_duplicat_from_localisation_H_traffic_light.csv'
# Total_cluster(input_CSV,output_CSV)
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
# labels="C:/visulisation_detection_final/lables.txt"
# import os 
# # Open the file in read mode
# with open(labels, 'r') as file:
#     # Read lines into a list, stripping the newline character
#     label = [line.strip() for line in file]
# print(label)



# for i in range(len(label)):
#     input_CSV=os.path.join("Localisation\csv_file\Traffic_sign","1_raw_points_"+str(label[i])+".csv")
#     output_CSV =os.path.join("Localisation\_2_csv_file\Traffic_sign","_2_unique_points_without_duplicat_from_localisation_H_"+str(label[i])+".csv")
#     if os.path.exists(input_CSV):
#         Total_cluster(input_CSV,output_CSV)


import os

# Specify the directory path
directory_path = 'Arlon_Localization/geoparquet'

# List all files in the directory
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]




for i in range(len(files)):
    input_CSV=os.path.join("Arlon_Localization\geoparquet",files[i])
    output_CSV =os.path.join("C:\PFE_Roadmap\Arlon_Localization\_2_csv","_2_unique_points_without_duplicat_from_localisation_H_"+files[i][12:])
    if os.path.exists(input_CSV):
        Total_cluster(input_CSV,output_CSV)
    