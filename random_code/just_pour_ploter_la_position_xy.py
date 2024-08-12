from PIL import Image
import matplotlib.pyplot as plt

def change_pixel_to_red(image_path, x, y,x1, y1):
    # Open the image
    img = Image.open(image_path)
    
    # Convert the image to RGB mode if it is not already
    img = img.convert("RGB")
    
    # Load the pixel data
    pixels = img.load()
    print(pixels)
    
    # Change the specified pixel to red
    pixels[x, y] = (255, 0, 0)
    pixels[x1, y1] = (255, 0, 0)
    # Get the new color of the specified pixel
    new_pixel_value = pixels[x, y]
    
    # Display the modified image
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()
    
    return new_pixel_value

# Example usage
image_path = "Sam/mask/VLg-91001-2023-02-11-083047000_0.png"  # Replace with the path to your image

x, y = 4432, 2350 # Replace with the coordinates of the pixel you want to change
x1, y1 = 4431, 1326
new_pixel_value = change_pixel_to_red(image_path, x, y,x1,y1)
print(f"The new color of the pixel at ({x}, {y}) is {new_pixel_value}")
