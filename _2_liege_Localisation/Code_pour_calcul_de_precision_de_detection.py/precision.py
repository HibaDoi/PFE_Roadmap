import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
def distance_2D(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

file_path="4_Localisation/LOB_final.csv"
df1 = pd.read_csv(file_path)
dff1=df1
df1=df1[["new_centroid_x","new_centroid_y"]]
# df1=df1[["x","$y"]]
columns = df1.columns



# Creating a new DataFrame with columns 'A' and 'C'

file_path="4_Localisation/GCP_final.csv"
df2 = pd.read_csv(file_path)

###################################################
clustering2 = DBSCAN(eps=5, min_samples=2).fit(df2[['xx', 'yy']])
df2['cluster2'] = clustering2.labels_
# Remove rows where cluster label is -1 (noise)
# filtered_filtered_df23 = filtered_filtered_df23[filtered_filtered_df23['cluster2'] != -1]

##################################################################################################################
# Separate the DataFrame into two parts
df2 = df2[df2['cluster2'] == -1]
################################################
df2.to_csv('4_Localisation/POT_red2.csv', index=False)
df2=df2[["xx","yy"]]
# Extraire les coordonnées XY des deux DataFrames
coords1 = df1.values
coords2 = df2.values

# Utiliser Nearest Neighbors pour trouver les points les plus proches
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(coords2)
distances, indices = nbrs.kneighbors(coords1)
print(distances)
input()
################
# Create an empty DataFrame with column names
df = pd.DataFrame(columns=['x', 'y'])
mean_distances = np.mean(distances, axis=0)
# Printing the mean distances
# print("Mean distances:", mean_distances)
# Flattening the 2D array to a 1D array
flattened_data = distances.flatten()
jj=[]
ii=[]
print("shape",len(flattened_data))
for i in range(len(flattened_data)):
    if flattened_data[i] < 1:
        j=flattened_data[i]**2
        u=flattened_data[i]
        jj.append(j)
        ii.append(u)
    else :
        print("more than 1m",flattened_data[i],i)
print("moyen",sum(ii)/len(ii))
print("RMSE",math.sqrt(sum(jj)/len(jj)))
input()
dff1["ecart"]=flattened_data
input(dff1)
# Creating the histogram
plt.hist(ii, bins=5, alpha=0.3, color='blue')  # Adjust 'bins' as needed for your full dataset

# Adding titles and labels
plt.title('Histogram of Sample Data')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Display the histogram
plt.show()
input("//////////////:")
# Creating a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(dff1['ecart'], dff1['mean_distance_to_camera'], color='blue')
plt.title('Scatter Plot of Mean Distance to Camera vs. Ecart')
plt.xlabel('Ecart')
plt.ylabel('Mean Distance to Camera')
plt.grid(True)
plt.show()
# # Adding data one by one
# for index in range(len(df2)):
#     df.loc[index] =[df1.loc[indices[index]]["x"].values[0],df1.loc[indices[index]]["$y"].values[0]]
   
# print(df)
# input([df.values[0],df2.values[0]])


# ###################

# actual_coords=df.values
# predicted_coords=df2.values


# def calculate_rmse_xy(actual, predicted):
#     actual = np.array(actual)
#     predicted = np.array(predicted)
#     differences = actual - predicted
#     squared_distances = np.sum(differences**2, axis=1)  # Summing squared differences across each row (x and y)
#     mean_squared_distances = np.mean(squared_distances)
#     rmse = np.sqrt(mean_squared_distances)
#     return rmse



# rmse = calculate_rmse_xy(actual_coords, predicted_coords)
# print("The RMSE for xy coordinates is:", rmse)
