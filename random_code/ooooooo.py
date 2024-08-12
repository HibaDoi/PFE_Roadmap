import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
# Original string with the values
df = pd.read_csv("sanity.csv")

result_df = df.melt(id_vars=['cluster'], value_vars=['Z1', 'Z2'])
clusters = result_df['cluster'].unique()  # Get unique clusters
Z=[]
for cluster in clusters:
    g=result_df[result_df['cluster'] == cluster]
    print(g[['value']])
    dbscan = DBSCAN(eps=0.1, min_samples=2)
    g['clusterZZt'] = dbscan.fit_predict(g[['value']])
    g = g[g['clusterZZt'] != -1]
    cluster_counts = g['clusterZZt'].value_counts()
    print(cluster_counts)
    most_frequent_cluster = cluster_counts.idxmax()
    gggg = g[g['clusterZZt'] == most_frequent_cluster]
    g1 = g[g['clusterZZt'] == most_frequent_cluster]
    gggg=gggg[['value']].mean().reset_index()
    print(gggg.values.tolist()[0][1])
    Z.append(gggg.values.tolist()[0][1])

input(Z)



    