import pandas as pd
import matplotlib.pyplot as plt
# Original string with the values
df = pd.read_csv("sanity.csv")
print(df)


# Use melt to reshape the DataFrame, repeating 'cluster' for each value
result_df = df.melt(id_vars=['cluster'], value_vars=['Z1', 'Z2'])



# Display the formatted output
clusters = result_df['cluster'].unique()  # Get unique clusters

for cluster in clusters:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax[0].hist(result_df[result_df['cluster'] == cluster]['value'], bins=10, color='blue', alpha=0.7)
    ax[0].set_title(f'Cluster {cluster} ')


    plt.show()