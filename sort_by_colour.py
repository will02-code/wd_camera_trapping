import os
from PIL import Image
import numpy as np

def get_average_color(image_path):
    """
    Returns the average (R, G, B) color of an image as a NumPy array of floats.
    """
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        np_img = np.array(img)
        # Compute mean over height and width, resulting in [R, G, B]
        avg_color = np.mean(np_img, axis=(0, 1))  
    return avg_color  # e.g., [R, G, B] floats

def sort_images_by_average_color(folder):
    # Get all image files
    image_files = [f for f in os.listdir(folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Compute average color for each image
    image_colors = []
    for filename in image_files:
        filepath = os.path.join(folder, filename)
        avg_col = get_average_color(filepath)
        image_colors.append((filename, avg_col))

    # Sort by (option 1): Red channel
    sorted_by_r = sorted(image_colors, key=lambda x: x[1][0])
    
    # Sort by (option 2): Green channel
    sorted_by_g = sorted(image_colors, key=lambda x: x[1][1])
    
    # Sort by (option 3): Blue channel
    sorted_by_b = sorted(image_colors, key=lambda x: x[1][2])
    
    # Sort by (option 4): Overall brightness (sum of R+G+B)
    sorted_by_brightness = sorted(image_colors, key=lambda x: sum(x[1]))

    return sorted_by_r, sorted_by_g, sorted_by_b, sorted_by_brightness

if __name__ == '__main__':
    folder_path = 'C:\\Users\\willo\\OneDrive - UNSW\\Documents\\Work\\CES\\Wild Deserts\\Image classification\\training\\model_training\\14_classes_b_plus_empty\\images\\test'
    s_r, s_g, s_b, s_brightness = sort_images_by_average_color(folder_path)
    
    print("Sorted by RED:\n", s_r)
    print("Sorted by GREEN:\n", s_g)
    print("Sorted by BLUE:\n", s_b)
    print("Sorted by Brightness:\n", s_brightness)
