import pandas as pd 
import os 
original="Arlon_Localization/Point_Cloud/_1_cropped_bounding_box_for_TS"
labels="C:/visulisation_detection_final/lables.txt"
labels_ex="C:/visulisation_detection_final/lables - Copie.txt"
# Open the file in read mode
from PIL import Image

def get_image_shape(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        return width, height
    
with open(labels, 'r') as file:
    # Read lines into a list, stripping the newline character
    label = [line.strip() for line in file]
with open(labels_ex, 'r') as file:
    # Read lines into a list, stripping the newline character
    label_ex = [line.strip() for line in file]
data= [r for r in os.listdir(original) if r.endswith(".txt")]
obj=[]
for item in data:
    prefix = item.split('__')[0] 
    obj.append(prefix[3:])
    

df = pd.DataFrame()
df["files"]=data


CLAS=[]
C=[]
Stat=[]
W=[]
H=[]
for item in data:
    with open(os.path.join(original,item[:-4]+".txt"), 'r') as file:
        line = file.readline()   
        
        stat ,clas =  line.split()
        CLAS.append(label_ex[int(label.index(clas))])
        C.append(clas)
        Stat.append(stat)
    files=os.path.join(original,item[:-4]+".jpg")
    width, height = get_image_shape(files)
    W.append(width)
    H.append(height)

df["object"] =obj
df["cl"]=C
df["classes"]=CLAS  
df["proba"] =Stat
df["W"]=W  
df["H"] =H
df["proba"]=df["proba"].astype(float)
df["object"]=df["object"].astype(int)
df = df[df["proba"]>= 0.7]
df = df[df["H"]>= 20]
df = df.groupby(['object', 'classes']).filter(lambda x: len(x) >= 2)

print(df)
df_grouped = df.groupby(['object','cl', 'classes'], as_index=False)['proba'].mean()
unique_values = df['object'].unique()

print(df_grouped)
print(len(unique_values))
import geopandas as gpd
df = pd.DataFrame(df_grouped)

# Set the CRS (for example, WGS 84)


# Save the GeoDataFrame as a GeoParquet file
df.to_parquet('Arlon_Localization/Point_Cloud/instance_type.parquet')