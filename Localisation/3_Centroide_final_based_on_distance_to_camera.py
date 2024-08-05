import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import geopandas as gpd
from shapely.geometry import Point
import os
# max distnace to camera 
def more_filtering(input,to_parquet,max_dist):
    # Load the data from the text file
    df = pd.read_csv(input)
    # Calculate new centroid positions based on existing points in each cluster
    new_centroids = df.groupby('cluster')[['centroid_xf', 'centroid_yf']].mean().reset_index()
    new_centroids.columns = ['cluster', 'new_centroid_x', 'new_centroid_y']
    # Merge the new centroids back to the original DataFrame
    df = df.merge(new_centroids, on='cluster')
    # Calculate the Euclidean distance to the new centroid for each point
    df['new_distance_to_centroid'] = np.sqrt((df['centroid_xf'] - df['new_centroid_x'])**2 + (df['centroid_yf'] - df['new_centroid_y'])**2)
    # Determine the number of points in each cluster
    cluster_counts = df['cluster'].value_counts()
    # Calculate points that would be removed by the filter
    df['remove'] = (df['mean_camera_distance'] > max_dist)
    removal_counts = df.groupby('cluster')['remove'].sum()
    # Reindex removal_counts to match cluster_counts and fill missing values with 0
    removal_counts = removal_counts.reindex(cluster_counts.index, fill_value=0)
    # Calculate clusters where not all points are removed
    valid_clusters = removal_counts[removal_counts < cluster_counts].index
    # Only apply the filter where the number of removals is less than the total points in the cluster
    filtered_df = df[(~df['cluster'].isin(valid_clusters) | ~df['remove'])]
    # Drop the temporary 'remove' column
    filtered_df = filtered_df.drop(columns='remove')

    #############################################################################################################################################
    filtered_df = filtered_df.drop(['new_centroid_x', 'new_centroid_y','new_distance_to_centroid'], axis=1)
    # Calculate new centroid positions based on existing points in each cluster
    new_centroids = filtered_df.groupby('cluster')[['centroid_xf', 'centroid_yf']].mean().reset_index()
    new_centroids.columns = ['cluster', 'new_centroid_x', 'new_centroid_y']
    # Merge the new centroids back to the original DataFrame
    filtered_df = filtered_df.merge(new_centroids, on='cluster')
    # Calculate the Euclidean distance to the new centroid for each point
    filtered_df['new_distance_to_centroid'] = np.sqrt((filtered_df['centroid_xf'] - filtered_df['new_centroid_x'])**2 + (filtered_df['centroid_yf'] - filtered_df['new_centroid_y'])**2)
    #############################################################################################################################################
    # Calculate points that would be removed by the filter
    filtered_df['remove'] = (filtered_df['new_distance_to_centroid'] > 0.5)
    removal_counts1 = filtered_df.groupby('cluster')['remove'].sum()
    # Reindex removal_counts1 to match cluster_counts and fill missing values with 0
    removal_counts1 = removal_counts1.reindex(cluster_counts.index, fill_value=0)
    # Calculate clusters where not all points are removed
    valid_clusters1 = removal_counts1[removal_counts1 < cluster_counts].index
    # Only apply the filter where the number of removals is less than the total points in the cluster
    filtered_filtered_df2 = filtered_df[(~filtered_df['cluster'].isin(valid_clusters1) | ~filtered_df['remove'])]
    Zdf=filtered_filtered_df2[['cluster',"Z1","Z2"]]
    print(filtered_filtered_df2)
    #######################################################################################################
    #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
    result_df = Zdf.melt(id_vars=['cluster'], value_vars=['Z1', 'Z2'])
    clusters = result_df['cluster'].unique()  # Get unique clusters
    Z=[]
    for cluster in clusters:
        Z_clustred = result_df[result_df['cluster'] == cluster].copy()
        Z_clustredjj = result_df[result_df['cluster'] == cluster].copy()
        dbscan = DBSCAN(eps=0.1, min_samples=2)
        Z_clustred.loc[:, 'clusterZZt'] = dbscan.fit_predict(Z_clustred[['value']])
        Z_clustred = Z_clustred[Z_clustred['clusterZZt'] != -1]
        cluster_counts = Z_clustred['clusterZZt'].value_counts()

        if cluster_counts.shape[0] !=0:
            most_frequent_cluster = cluster_counts.idxmax()
            Z_filtred = Z_clustred[Z_clustred['clusterZZt'] == most_frequent_cluster]
            Z_filtred=Z_filtred[['value']].mean().reset_index()

        #####################################################################################################")
            Z.append(Z_filtred.values.tolist()[0][1])
        else:
            Z.append(Z_clustredjj['value'].mean())
            
    #iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii
    #####################################################################################################
    filtered_filtered_df23 = filtered_filtered_df2.groupby(['cluster']).agg(
                    mean_distance_to_centroid=('new_distance_to_centroid', 'mean'),
                    new_centroid_x=('centroid_xf', 'mean'),  # Replace 'centroid_xf' with your x-coordinate column
                    new_centroid_y=('centroid_yf', 'mean'),   # Replace 'centroid_yf' with your y-coordinate column
                    mean_distance_to_camera=('mean_camera_distance', 'mean'),
                    H1=('H1', 'mean'),  # Replace 'centroid_xf' with your x-coordinate column
                    H2=('H2', 'mean'), 
                ).reset_index()
    filtered_filtered_df23['Z_final']=Z
    ################################################################################################################"
    # # Arrondir les colonnes 'x' et 'y' à quatre chiffres après la virgule
    filtered_filtered_df23['new_centroid_x'] = filtered_filtered_df23['new_centroid_x'].round(4)
    filtered_filtered_df23['new_centroid_y'] = filtered_filtered_df23['new_centroid_y'].round(4)
    # Supprimer les doublons
    # Apply DBSCAN
    clustering2 = DBSCAN(eps=5, min_samples=2).fit(filtered_filtered_df23[['new_centroid_x', 'new_centroid_y']])
    filtered_filtered_df23['cluster2'] = clustering2.labels_
    # Remove rows where cluster label is -1 (noise)
    # filtered_filtered_df23 = filtered_filtered_df23[filtered_filtered_df23['cluster2'] != -1]
    ##################################################################################################################
    # Separate the DataFrame into two parts
    df_negative_ones = filtered_filtered_df23[filtered_filtered_df23['cluster2'] == -1]
    df_not_negative_ones = filtered_filtered_df23[filtered_filtered_df23['cluster2'] != -1]
    # Find the index of the minimum 'mean_distance_to_camera' for each 'cluster2' group
    idx = df_not_negative_ones.groupby('cluster2')['mean_distance_to_camera'].idxmin()
    # Select rows with these indices
    min_distance_df = filtered_filtered_df23.loc[idx]
    # Concatenate rows with 'cluster2' equal to -1 with the rows having the minimum 'mean_distance_to_camera'
    final_df = pd.concat([df_negative_ones, min_distance_df]).reset_index(drop=True)
    #########################################################################################################
    # final_df.to_csv(output, index=False)
    geometry = [Point(xy) for xy in zip(final_df['new_centroid_x'],final_df['new_centroid_y'])]
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(final_df, geometry=geometry)
    # Set a coordinate reference system (CRS) if necessary, e.g., WGS84
    gdf.set_crs(epsg=31370, inplace=True)
    # Save to a shapefile
    gdf.to_parquet(to_parquet)

output_dir= "Localisation/_3_final_point/Traffic_sign"
input_dir="Localisation/_2_csv_file/Traffic_sign"
print('////////////////////////:')

files = os.listdir(input_dir)
csv_files = [file for file in files if file.endswith('.csv')]
print(files)
for file in csv_files:
    input=os.path.join(input_dir,file)
    to_parquet=os.path.join(output_dir,"Final__"+file[54:-4]+"__.geoparquet")
    # print(to_parquet)
    more_filtering(input,to_parquet,25)

# input='Arlon_Localization/_2_csv/_2_unique_points_without_duplicat_from_localisation_H__single_lamppost.csv'
# to_parquet='Arlon_Localization/_3_final_Arlon/Finalocalisation_H__single_lamppost.geoparquet'
# more_filtering(input,to_parquet,25)