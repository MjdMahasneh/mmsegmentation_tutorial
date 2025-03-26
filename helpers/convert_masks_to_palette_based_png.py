from PIL import Image
import numpy as np
import os

# Define your palette (same order as CLASSES)
palette = [
    (255, 0, 255),   # Wall
    (255, 0, 0),     # Door
    (255, 255, 0),   # Roof
    (0, 0, 255),     # Window
    (255, 255, 255)  # Background
]

# Paths
data_root = './dataset'
ann_dir = 'masks'
output_dir = 'output'  # Save converted masks here

os.makedirs(os.path.join(data_root, output_dir), exist_ok=True)

# Convert each mask
for filename in os.listdir(os.path.join(data_root, ann_dir)):
    if filename.endswith('.png'):
        # Load the RGB mask
        rgb_mask = Image.open(os.path.join(data_root, ann_dir, filename)).convert('RGB')
        rgb_array = np.array(rgb_mask)

        # Create a label map with the same shape as the mask
        label_map = np.zeros((rgb_array.shape[0], rgb_array.shape[1]), dtype=np.uint8)

        # Map RGB values to class indices
        for idx, rgb in enumerate(palette):
            matches = np.all(rgb_array == np.array(rgb), axis=-1)
            label_map[matches] = idx  # Assign label index

        # Convert to indexed PNG with the correct palette
        label_img = Image.fromarray(label_map, mode='P')
        label_img.putpalette(np.array(palette, dtype=np.uint8).flatten())  # Apply color palette

        # Save as an indexed PNG
        label_img.save(os.path.join(data_root, output_dir, filename))
