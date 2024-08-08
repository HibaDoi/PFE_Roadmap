import pandas as pd 
import os 
original="Arlon_Localization/Point_Cloud/cropped_image_traffic_light"

# Open the file in read mode
from PIL import Image

def get_image_shape(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height
    

data= [r for r in os.listdir(original) if r.endswith(".txt")]

obj=[]
for item in data:
    prefix = item.split('__')[0] 
    obj.append(prefix[3:])
    

df = pd.DataFrame()
df["files"]=data



C=[]
Stat=[]
W=[]
H=[]
for item in data:
    with open(os.path.join(original,item[:-4]+".txt"), 'r') as file:
        line = file.readline()   

        stat  =  line.split()[0]
        clas =line.split()[1]
        C.append(clas)
        Stat.append(stat)
    files=os.path.join(original,item[:-4]+".jpg")
    width, height = get_image_shape(files)
    W.append(width)
    H.append(height)

df["object"] =obj
df["cl"]=C
df["proba"] =Stat
df["W"]=W  
df["H"] =H
df["proba"]=df["proba"].astype(float)
df["object"]=df["object"].astype(int)
df = df[df["proba"]>= 0.999]
df = df[df["H"]>= 20]
df = df[df['cl'] == 'Traffic_Light']
df = df.groupby(['object', 'cl']).filter(lambda x: len(x) >= 2)


df.to_csv('filtered_data.csv', index=False)


df = df.groupby(['object', 'cl'], as_index=False).agg(
    proba_mean=('proba', 'mean'),
    count=('proba', 'size')
)

df=df[df['count']>=3]

print(df)



import geopandas as gpd
df = pd.DataFrame(df)

# Set the CRS (for example, WGS 84)


# Save the GeoDataFrame as a GeoParquet file
df.to_parquet('Arlon_Localization/Point_Cloud/instance_type_Traffic_light.parquet')