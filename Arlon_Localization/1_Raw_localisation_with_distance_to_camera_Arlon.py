from utils_localisation_Arlon import *
import json
from sklearn.cluster import DBSCAN
import pandas as pd
img_width=7040
img_height=3520
camera_info_file="Arlon_Localization/RPY.csv"
#this should contain .json 
directory_path = 'C:/visulisation_detection_final/Arlon_Lamppost/json_single_adding_hauteur'
file_path = 'Arlon_Localization/geoparquet/1_raw_points.geoparquet'
def XYlocation(camera_info_file,directory_path,file_path):
    u=traverse_directory_in_groups(directory_path)
    camera_info=parse_image_info(camera_info_file)
    for j in range(len(u)-4) :
        print("________________________________Premier 4 points__________________________________________")
        print(str(u[j]))
        l=[str(u[j]) ,str(u[j+1]),str(u[j+2]),str(u[j+3])]
        V_total=[]
        V_total2=[]
        ab_total=[]
        GCP_total=[]
        G_total=[]
        C_total=[[0,0],
                [0,0],
                [0,0],
                [0,0]]
        for i in range(4):
            json_filemane=l[i][:-5]+".json"
            with open(json_filemane, 'r') as file:
                data_xy = json.load(file)
            parts = l[i].split('\\')
            target_entry = next((entry for entry in camera_info if entry['filename'] == parts[-1][:-5]+".jpg"), None)
            ab=[]
            GCP=[]
            G=[]
            V=[]
            V2=[]
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
                    xt=data_xy[j]["xyt"][0]
                    yt=data_xy[j]["xyt"][1]
                    ii=[lon,lat,height,roll,pitch,yaw ,x,y,xt,yt]
                    gisement,angle_vertical,Camera_Coordinate_Lambert_72,Par_droit=ObjectCoordinat(ii[3:6],ii[0:3],(ii[6], ii[7]),img_width,img_height)
                    _,angle_vertical2,_,_=ObjectCoordinat(ii[3:6],ii[0:3],(ii[8], ii[9]),img_width,img_height)
                    GCP.append([gisement,Camera_Coordinate_Lambert_72,Par_droit])
                    ab.append(Par_droit)
                    G.append(gisement)
                    C_total[i]=Camera_Coordinate_Lambert_72
                    V.append(angle_vertical)
                    V2.append(angle_vertical2)
                    # Calculate intersections    
            else:
                print(f"Filename  not found in the data.")
            V_total.append(V)
            V_total2.append(V2)
            ab_total.append(ab)
            GCP_total.append(GCP)
            G_total.append(G)

        uu=find_intersections(C_total, G_total,ab_total,V_total,V_total2)
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
        df['V1'] = [uu[i][3] for i in range(len(uu))]
        df['V1_'] = [uu[i][5] for i in range(len(uu))]
        df['camera2'] = [uu[i][2][0] for i in range(len(uu))]
        df['lob2'] = [uu[i][2][2] for i in range(len(uu))]
        df['V2'] = [uu[i][4] for i in range(len(uu))]
        df['V2_'] = [uu[i][6] for i in range(len(uu))]
        df['xcamera1coo'] = [C_total[uu[i][1][0]][0] for i in range(len(uu))]
        df['ycamera1coo'] = [C_total[uu[i][1][0]][1] for i in range(len(uu))]
        df['zcamera1coo'] = [C_total[uu[i][1][0]][2] for i in range(len(uu))]
        df['xcamera2coo'] = [C_total[uu[i][2][0]][0] for i in range(len(uu))]
        df['ycamera2coo'] = [C_total[uu[i][2][0]][1] for i in range(len(uu))]
        df['zcamera2coo'] = [C_total[uu[i][2][0]][2] for i in range(len(uu))]
        df_filtered = df[df['cluster'] != -1]
        if not df_filtered.empty :
            # Affichage du DataFrame filtré
            gg=df_filtered.sort_values(by='cluster')
            gg=gg.reset_index(drop=True)
            # Calcul des centroïdes
            centroid_df = gg.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).reset_index()
            centroid_df.columns = ['cluster', 'centroid_x', 'centroid_y']
            # Fonction pour calculer la distance Euclidienne
            # Ajouter les centroïdes au DataFrame original
            gg = gg.merge(centroid_df, on='cluster', how='left')
            # Calculer la distance de chaque point à son centroïde
            gg['distance_to_centroid'] = gg.apply(lambda row: calculate_distance(row['x'], row['y'], row['centroid_x'], row['centroid_y']), axis=1)
            gg['distance_to_camera1'] = gg.apply(lambda row: calculate_distance(row['x'], row['y'], row['xcamera1coo'], row['ycamera1coo']), axis=1)
            gg['distance_to_camera2'] = gg.apply(lambda row: calculate_distance(row['x'], row['y'], row['xcamera2coo'], row['ycamera2coo']), axis=1)
            gg['Z1'] = gg.apply(lambda row: calculate_Z(row['distance_to_camera1'], row['V1'], row['zcamera1coo']), axis=1)
            gg['Z2'] = gg.apply(lambda row: calculate_Z(row['distance_to_camera2'], row['V2'], row['zcamera2coo']), axis=1)
            gg['H1'] = gg.apply(lambda row: calculate_H(row['distance_to_camera1'], row['V1']-row['V1_']), axis=1)
            gg['H2'] = gg.apply(lambda row: calculate_H(row['distance_to_camera2'], row['V2']-row['V2_']), axis=1)
            # Trier les données par cluster, combinaison de caméras, et distance, puis éliminer les doublons
            # Sélectionner toutes les lignes mais uniquement certaines colonnes
            ggn = gg.loc[:, ['cluster','camera1','lob1','distance_to_camera1', 'camera2','lob2','distance_to_camera2']]
            gga=gg.loc[:, ['cluster','camera1','lob1','distance_to_camera1']]
            print(gga)
            ggb=gg.loc[:, ['cluster','camera2','lob2','distance_to_camera2']]
            print(ggb)
            gga.columns = ['cluster', 'camera', 'lob', 'distance_to_camera']
            ggb.columns = ['cluster', 'camera', 'lob', 'distance_to_camera']
            result = pd.concat([gga, ggb])
            # Trier le DataFrame final par cluster puis réinitialiser l'index pour nettoyer l'affichage
            result_sorted = result.sort_values(by=['cluster', 'camera', 'lob']).reset_index(drop=True)
            # Affichage du DataFrame final
            # Group by 'cluster', 'camera', and 'lob', then calculate the mean distance
            result1 = result_sorted.groupby(['cluster', 'camera', 'lob']).agg(mean_distance=('distance_to_camera', 'mean')).reset_index()
            # Display the resulting DataFrame
            # Sort by 'camera', 'lob', and 'mean_distance' (ascending)
            sorted_df = result1.sort_values(by=['camera', 'lob', 'mean_distance'], ascending=[True, True,   False])
            # Drop duplicates while keeping the entry with the lowest distance for each 'camera-lob' combination
            final_df = sorted_df.drop_duplicates(subset=['camera', 'lob'], keep='first')
            final_df =final_df.sort_values(by=['cluster', 'camera', 'lob'])
            # Display the resulting DataFrame
            # Finding unique clusters in df1
            unique_clusters_df11 = final_df['cluster'].unique()
            # Filter df2 to only include rows with clusters that are unique in df1
            gg = gg[gg['cluster'].isin(unique_clusters_df11)]
            # (1) Find clusters in table2 with only one entry
            single_entry_clusters = gg['cluster'].value_counts()[final_df['cluster'].value_counts() == 1].index
            gg = gg[~gg['cluster'].isin(single_entry_clusters)]
            valid_combinations = final_df.groupby('cluster').apply(lambda df: set(zip(df['camera'], df['lob']))).to_dict()
            gg = gg[
            gg.apply(lambda row: (row['camera1'], row['lob1']) in valid_combinations.get(row['cluster'], set()) and (row['camera2'], row['lob2']) in valid_combinations.get(row['cluster'], set()), axis=1)]
            # (1) Count entries in each cluster within table1
            cluster_counts = gg['cluster'].value_counts()
            # Identify clusters with only one entry
            clusters_with_single_entry = cluster_counts[cluster_counts == 1].index
            # Drop rows from table1 where cluster has only one entry
            gg = gg[~gg['cluster'].isin(clusters_with_single_entry)]
            # Calcul des centroïdes
            centroid_df = gg.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).reset_index()
            centroid_df.columns = ['cluster', 'centroid_xf', 'centroid_yf']
            # Ajouter les centroïdes au DataFrame original
            gg = gg.merge(centroid_df, on='cluster', how='left')
            # Calculer la distance de chaque point à son centroïde
            gg['distance_to_centroidf'] = gg.apply(lambda row: calculate_distance(row['x'], row['y'], row['centroid_xf'], row['centroid_yf']), axis=1)
            # Trier les données par cluster, combinaison de caméras, et distance, puis éliminer les doublons
            print(gg)
            print("###########################################")
            ggg = gg.loc[:, ['cluster', 'centroid_xf', 'centroid_yf','distance_to_centroidf','distance_to_camera1','distance_to_camera2',"Z1","Z2","H1","H2"]]
            # Group by 'cluster', 'centroid_x', 'centroid_y', then calculate the mean of 'distance_to_centroid'
            grouped_df = ggg.groupby(['cluster', 'centroid_xf', 'centroid_yf']).agg(
                mean_distance_to_centroid=('distance_to_centroidf', 'mean'),
                mean_camera_distance1=('distance_to_camera1', 'mean') ,
                mean_camera_distance2=('distance_to_camera2', 'mean'),
                Z1=("Z1", 'mean') ,
                Z2=("Z2", 'mean'),
                H1=("H1", 'mean') ,
                H2=("H2", 'mean'),
            ).reset_index()

            # Print the resulting DataFrame
            grouped_df['mean_camera_distance'] = grouped_df[['mean_camera_distance1', 'mean_camera_distance1']].mean(axis=1)

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















    