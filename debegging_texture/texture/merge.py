import json

def merge_multisurfaces_with_textures(cityjson_data, multisurface_ids):
    # Initialize the CompositeSurface structure
    composite_surface = {
        "type": "CompositeSurface",
        "lod": 2,
        "boundaries": [],
        "texture": {"values": []}
    }
    
    # Get the appearance section (textures)
    appearance = cityjson_data.get("appearance", {})
    
    # Iterate over the given MultiSurface IDs and add their geometry and texture information
    for ms_id in multisurface_ids:
        for geometry in cityjson_data["CityObjects"][ms_id]["geometry"]:
            if geometry["type"] == "MultiSurface":
                # Add the geometry boundaries to the CompositeSurface
                composite_surface["boundaries"].append(geometry["boundaries"])
                
                # Find the corresponding texture index and add it to the texture values
                if "texture" in geometry:
                    composite_surface["texture"]["values"].append(geometry["texture"]["default"]["values"])

    # Add the new CompositeSurface to the CityJSON data
    new_geom_id = "CompositeSurface_merged"
    cityjson_data["CityObjects"][new_geom_id] = {
        "type": "Building",  # Adjust this type if necessary
        "geometry": [composite_surface]
    }
    
    return cityjson_data

# Load your CityJSON file
with open("texture/okkk.json", "r") as file:
    cityjson_data = json.load(file)

# List of MultiSurface IDs you want to merge
multisurface_ids = ["Circle002", "Line006", "s","sipka"]  # Replace with actual IDs

# Merge the MultiSurfaces with their textures
merged_cityjson = merge_multisurfaces_with_textures(cityjson_data, multisurface_ids)

# Save the modified CityJSON file
with open("texture/okkk_merged.json", "w") as file:
    json.dump(merged_cityjson, file, indent=2)

print("MultiSurfaces merged successfully.")
