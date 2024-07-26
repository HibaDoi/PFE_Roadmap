from utils_Final import *
import json
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

camera_info_file="Localisation/Camera_parameter.txt"
directory_path = 'Json+mask'
file_path = 'Localisation\csv_file\_1_raw_points_from_localisation.csv'
# input_CSV=file_path
# output_CSV='4_Localisation/localisation/final_traffic_light.csv'
def XYlocation(camera_info_file,directory_path,file_path):
    print("..................")
    u=traverse_directory_in_groups(directory_path)
    print(u)
    camera_info=parse_image_info(camera_info_file)
    harimna=[]
    for j in range(len(u)-4) :
        print("________________________________Premier 4 points__________________________________________")
        print(str(u[j]))
        print(str(u[j+1]))
        print(str(u[j+2]))
        print(str(u[j+3]))
        l=[str(u[j]) ,str(u[j+1]),str(u[j+2]),str(u[j+3])]

        ab_total=[]
        GCP_total=[]
        G_total=[]
        C_total=[[0,0],
                [0,0],
                [0,0],
                [0,0]]
        for i in range(4):
            filename=l[i]
            json_filemane=l[i]
            # print(json_filemane)
            # input("dddddddddddddd")
            with open(json_filemane, 'r') as file:
                data_xy = json.load(file)
            parts = l[i].split('\\')
            # print(parts)
            # input("dddddddddddddd")
            target_entry = next((entry for entry in camera_info if entry['filename'] == parts[-1][:-5]+".jpg"), None)
            ab=[]
            GCP=[]
            G=[]
            if target_entry:
                for j in range(len(data_xy)):
                    lon = target_entry['imj_lon']  # Assuming width is pitch for this example
                    lat = target_entry['imj_lat']  # Assuming height is yaw for this example
                    height = target_entry['imj_height']  # Assuming width is pitch for this example
                    roll = target_entry['imj_roll']  # Assuming height is yaw for this example
                    pitch = target_entry['imj_pitch']  # Assuming width is pitch for this example
                    yaw = target_entry['imj_yaw']  # Assuming height is yaw for this example
                    
                    x=data_xy[j]["xy"][0]
                    y=data_xy[j]["xy"][1]
                    ii=[lon,lat,height,roll,pitch,yaw ,x,y]
                    gisement,Camera_Coordinate_Lambert_72,Par_droit=ObjectCoordinat(ii[3:6],ii[0:3],(ii[6], ii[7]))
                    GCP.append([gisement,Camera_Coordinate_Lambert_72,Par_droit])
                    ab.append(Par_droit)
                    G.append(gisement)
                    C_total[i]=Camera_Coordinate_Lambert_72
                    # Calculate intersections    
            else:
                print(f"Filename  not found in the data.")
        
            print("__________________________getting info about point_",i,"_______________________________________")
            
            ab_total.append(ab)
            GCP_total.append(GCP)
            G_total.append(G)
        x_values = [coord[0] for coord in C_total]
        y_values = [coord[1] for coord in C_total]
        uu=find_intersections2(C_total, G_total,ab_total)
        ff=[]
        for i in range(len(uu)):
            ff.append(uu[i][0])
        df = pd.DataFrame()
        if len(ff) == 1:
            print("1")
            continue
        elif len(ff) == 0:
            print('0')
            continue
        else:
            clustering = DBSCAN(eps=1, min_samples=2).fit(ff)
            df['cluster'] = clustering.labels_

        
        
        # Ajout de colonnes
        
        df['x'] = [uu[i][0][0] for i in range(len(uu))]
        df['y'] = [uu[i][0][1] for i in range(len(uu))]
        df['camera1'] = [uu[i][1][0] for i in range(len(uu))]
        df['lob1'] = [uu[i][1][2] for i in range(len(uu))]
        df['camera2'] = [uu[i][2][0] for i in range(len(uu))]
        df['lob2'] = [uu[i][2][2] for i in range(len(uu))]
        df['xcamera1coo'] = [C_total[uu[i][1][0]][0] for i in range(len(uu))]
        df['ycamera1coo'] = [C_total[uu[i][1][0]][1] for i in range(len(uu))]
        df['xcamera2coo'] = [C_total[uu[i][2][0]][0] for i in range(len(uu))]
        df['ycamera2coo'] = [C_total[uu[i][2][0]][1] for i in range(len(uu))]
        df_filtered = df[df['cluster'] != -1]
        if not df_filtered.empty :
            # Affichage du DataFrame filtré
            gg=df_filtered.sort_values(by='cluster')
            gg=gg.reset_index(drop=True)
            print(gg)
            #input()

        
            # Calcul des centroïdes
            centroid_df = gg.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).reset_index()
            centroid_df.columns = ['cluster', 'centroid_x', 'centroid_y']
            # Fonction pour calculer la distance Euclidienne
            # Ajouter les centroïdes au DataFrame original
            gg = gg.merge(centroid_df, on='cluster', how='left')
            # Calculer la distance de chaque point à son centroïde
            gg['distance_to_centroid'] = gg.apply(lambda row: calculate_distance(row['x'], row['y'], row['centroid_x'], row['centroid_y']), axis=1)
            print(gg)
            #input()

            gg['distance_to_camera1'] = gg.apply(lambda row: calculate_distance(row['x'], row['y'], row['xcamera1coo'], row['ycamera1coo']), axis=1)
            gg['distance_to_camera2'] = gg.apply(lambda row: calculate_distance(row['x'], row['y'], row['xcamera2coo'], row['ycamera2coo']), axis=1)
            # Trier les données par cluster, combinaison de caméras, et distance, puis éliminer les doublons
            print(gg)
            #input()


            # Sélectionner toutes les lignes mais uniquement certaines colonnes
            ggn = gg.loc[:, ['cluster','camera1','lob1','distance_to_camera1', 'camera2','lob2','distance_to_camera2']]

            print(ggn)
            #input()

            gga=gg.loc[:, ['cluster','camera1','lob1','distance_to_camera1']]
            print(gga)
            ggb=gg.loc[:, ['cluster','camera2','lob2','distance_to_camera2']]
            print(ggb)
            #input()
            gga.columns = ['cluster', 'camera', 'lob', 'distance_to_camera']
            ggb.columns = ['cluster', 'camera', 'lob', 'distance_to_camera']
            result = pd.concat([gga, ggb])

            # Trier le DataFrame final par cluster puis réinitialiser l'index pour nettoyer l'affichage
            result_sorted = result.sort_values(by=['cluster', 'camera', 'lob']).reset_index(drop=True)

            # Affichage du DataFrame final
            print(result_sorted)
            #input("____________________________")

            # Group by 'cluster', 'camera', and 'lob', then calculate the mean distance
            result1 = result_sorted.groupby(['cluster', 'camera', 'lob']).agg(mean_distance=('distance_to_camera', 'mean')).reset_index()

            # Display the resulting DataFrame
            print(result1)
            #input(",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
            # Sort by 'camera', 'lob', and 'mean_distance' (ascending)
            sorted_df = result1.sort_values(by=['camera', 'lob', 'mean_distance'], ascending=[True, True,   False])
            print(sorted_df)
            #input('______________lets see___________________')
            # Drop duplicates while keeping the entry with the lowest distance for each 'camera-lob' combination
            final_df = sorted_df.drop_duplicates(subset=['camera', 'lob'], keep='first')
            print(final_df)
            final_df =final_df.sort_values(by=['cluster', 'camera', 'lob'])

            # Display the resulting DataFrame
            print(final_df)
            #input('ooooooooooooooooooooooooooo')
            # Finding unique clusters in df1
            unique_clusters_df11 = final_df['cluster'].unique()
            print(unique_clusters_df11)
            # Filter df2 to only include rows with clusters that are unique in df1
            gg = gg[gg['cluster'].isin(unique_clusters_df11)]

            print(gg)
            #input("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            # (1) Find clusters in table2 with only one entry
            single_entry_clusters = gg['cluster'].value_counts()[final_df['cluster'].value_counts() == 1].index
            gg = gg[~gg['cluster'].isin(single_entry_clusters)]
            print(gg)
            print(single_entry_clusters)
            #input("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
            valid_combinations = final_df.groupby('cluster').apply(lambda df: set(zip(df['camera'], df['lob']))).to_dict()
            print(valid_combinations)
            #input('jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj')
            gg = gg[
            gg.apply(lambda row: (row['camera1'], row['lob1']) in valid_combinations.get(row['cluster'], set()) and
                                    (row['camera2'], row['lob2']) in valid_combinations.get(row['cluster'], set()), axis=1)
        ]
            print(gg)
            #input('blablablablabala')
            # (1) Count entries in each cluster within table1
            cluster_counts = gg['cluster'].value_counts()

            # Identify clusters with only one entry
            clusters_with_single_entry = cluster_counts[cluster_counts == 1].index

            # Drop rows from table1 where cluster has only one entry
            gg = gg[~gg['cluster'].isin(clusters_with_single_entry)]
            print(gg)
            #input("##########################")


            # Calcul des centroïdes
            centroid_df = gg.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).reset_index()
            centroid_df.columns = ['cluster', 'centroid_xf', 'centroid_yf']

            # Ajouter les centroïdes au DataFrame original
            gg = gg.merge(centroid_df, on='cluster', how='left')
            print(gg)
            #input()
            # Calculer la distance de chaque point à son centroïde
            gg['distance_to_centroidf'] = gg.apply(lambda row: calculate_distance(row['x'], row['y'], row['centroid_xf'], row['centroid_yf']), axis=1)
            # Trier les données par cluster, combinaison de caméras, et distance, puis éliminer les doublons
            print(gg)
            #input("__________________88888888888888888888888888888888____________________")
            ggg = gg.loc[:, ['cluster', 'centroid_xf', 'centroid_yf','distance_to_centroidf']]

            # Group by 'cluster', 'centroid_x', 'centroid_y', then calculate the mean of 'distance_to_centroid'
            grouped_df = ggg.groupby(['cluster', 'centroid_xf', 'centroid_yf']).agg(
                mean_distance_to_centroid=('distance_to_centroidf', 'mean')
            ).reset_index()

            # Print the resulting DataFrame
            print(grouped_df)

            #input("__________________88888888111111111111111118888888____________________")
            

            # Specify the file path (change this path as needed for your environment)
            

            # Check if the file exists. If it does, append without headers. If not, write with headers.
            try:
                # Attempt to read the file to check if it exists
                pd.read_csv(file_path)
                # If it exists, append without writing headers
                grouped_df.to_csv(file_path, mode='a', header=False, index=False)
            except FileNotFoundError:
                # If the file does not exist, write with headers
                grouped_df.to_csv(file_path, mode='w', header=True, index=False)
        else:
            print("No Intersection Here")
    
XYlocation(camera_info_file,directory_path,file_path)
# Total_cluster(input_CSV,output_CSV)














    