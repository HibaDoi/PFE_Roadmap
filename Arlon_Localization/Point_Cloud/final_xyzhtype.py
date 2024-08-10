import pandas as pd 
import os 
import numpy as np
import geopandas as gpd

df_ts = pd.read_parquet('Arlon_Localization/Point_Cloud/type_position_files/instance_type_traffic_sign.parquet')
df_tl=pd.read_parquet('Arlon_Localization/Point_Cloud/type_position_files/instance_type_Traffic_light.parquet')
gdf = pd.read_parquet('Arlon_Localization/Point_Cloud/type_position_files/instance_xyzh_TS_TL.geoparquet')
# Inner join (only matching objects)
df_ts = df_ts[df_ts['cl'] != 'A51']
df_ts = df_ts[df_ts['proba'] >= 0.85]
result = df_ts.groupby('object').agg({
    'cl': lambda x: ', '.join(x.unique()),
    'classes': lambda x: ', '.join(x.unique()),
    'proba': lambda x: ', '.join(map(str, x))
})


merged_outer= pd.merge(result, gdf,left_on='object', right_on='segment_id', how='outer')
merged_outer=merged_outer[['cl','proba','segment_id','X','Y','Z','height']]
merged_outer= pd.merge(df_tl, merged_outer,left_on='object', right_on='segment_id', how='outer')
merged_outer = merged_outer.sort_values(by='segment_id')
merged_outer=merged_outer[['segment_id','cl_x','cl_y','proba','X','Y','Z','height']]
merged_outer = merged_outer.dropna(subset=['cl_x', 'cl_y'], how='all')
merged_outer['final_cl'] = merged_outer['cl_x'].fillna(merged_outer['cl_y'])
merged_outer=merged_outer[['segment_id','final_cl','X','Y','Z','height']]
print(merged_outer)
print(merged_outer.shape)
# Convert the DataFrame to a GeoDataFrame by adding a geometry column
gdff = gpd.GeoDataFrame(merged_outer, geometry=gpd.points_from_xy(merged_outer['X'], merged_outer['Y']))

# Set the CRS (Coordinate Reference System) to WGS 84 (EPSG:4326)
gdff.set_crs(epsg=31370, inplace=True)
gdff.to_parquet('Arlon_Localization/Point_Cloud/type_position_files/type_xyzh_tls.parquet')

# Display the GeoDataFrame
print(gdff)
input()

# print(result)
# input()
# merged_inner = pd.merge(df_ts, df_tl, on='object', how='left')

# # Assuming your DataFrame is named df
# column_names = merged_inner.columns

# ff=merged_inner[['object','cl_x','cl_y','proba','proba_mean']]
# ff['final_obj'] = np.where(ff['cl_y'].notna(), ff['cl_y'], ff['cl_x'])
# ff = ff[ff['final_obj'] != 'A51']
# ff = ff[ff['proba'] >=0.85]
# print(ff[['object','final_obj','proba']])
# # Group by 'object' and concatenate unique 'final_obj' values
# result = ff.groupby('object').agg({
#     'final_obj': lambda x: ', '.join(x.unique()),
#     'proba': lambda x: ', '.join(map(str, x))
# })


# # Reset the index if you want to convert the Series back to a DataFrame

# result = result.reset_index()
# result['segment_id']=result['object']
# # Display the result
# print(result)
# print(gdf)
# merged_inner = pd.merge(result, gdf, on='segment_id', how='inner')
# print(merged_inner)
# input()