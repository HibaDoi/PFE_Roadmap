import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import geopandas as gpd
from shapely.geometry import Point
def clustering(csv_avant_clustering,csv_apres_clustering):
    # Load the data from the text file
    data = pd.read_csv(csv_avant_clustering, header=None, delim_whitespace=True, names=['centroid_xf', 'centroid_yf'])
    data1 = pd.read_csv(csv_avant_clustering)
    
    data=data[1:]
    data=data.values.tolist()
    
    if data!=[]:
        for i in range(len(data)):
            data[i]=[float(value) for value in data[i][0].split(",")]
        
        
        for i in range(len(data)):
            data[i]=data[i][1:3]
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=1)
        clusters = dbscan.fit_predict(data)
        
        # Calculate centroids of each cluster
        centroids = []
        for label in np.unique(clusters):
            members = data1[clusters == label]
            centroid = members.mean(axis=0)
            centroids.append([centroid["centroid_xf"],centroid["centroid_yf"]])
        
        # Create a new DataFrame for centroids
        centroid_df = pd.DataFrame(centroids, columns=['centroid_x', 'centroid_y'])
        
        # # Save centroids to a new text file
        # centroid_df.to_csv(csv_apres_clustering, index=False, header=False, sep=' ')

        # print("Centroids have been calculated and saved .")
        ################################################################################
        # Create a list of shapely Point objects
        geometry = [Point(xy) for xy in zip(centroid_df['centroid_x'],centroid_df['centroid_y'])]

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame(centroid_df, geometry=geometry)

        # Set a coordinate reference system (CRS) if necessary, e.g., WGS84
        gdf.set_crs(epsg=31370, inplace=True)

        # Save to a shapefile
        gdf.to_file(csv_apres_clustering)

# dir_input="localisation/Traffic_sign_avant_clustering"
# dir_output="localisation/Traffic_sign_apres_clusteringt_shp"
# os.mkdir(dir_output)
# for j,i in enumerate(os.listdir(dir_input)):
#     print(i)
#     csv_avant_clustering=os.path.join(dir_input,i)
#     shp_apres_clustering=os.path.join(dir_output,"centroide_"+i)[:-4]+".shp"
#     clustering(csv_avant_clustering,shp_apres_clustering)
clustering("localisation/Bus_stop_archive/Bus_Stop_Final.csv","localisation/Bus_stop_archive/Bus_Stop_Final.shp")