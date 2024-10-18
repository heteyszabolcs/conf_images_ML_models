# data wrangling
import numpy as np
import os
# image processing
import tifffile
from PIL import Image
from pathlib import Path
from src.image_processing import process_image

def prepare_data(base_folder, target_size, normalize=True):
    """
    Prepare image data and labels for random forest from folders of JPG images.

    Args:
    base_folder (str): Path to the base folder containing 'A' and 'B' subfolders.
    target_size (tuple): Desired size of the output images (height, width).
    normalize (bool): If True, normalize pixel values to [0, 1].

    Returns:
    tuple: (X, y) where X is a numpy array of flattened image data and y is a numpy array of labels.
    """

    # Check if base folder exists
    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"The base folder {base_folder} does not exist.")

    # Check if 'A' and 'B' subfolders exist
    folder_a = os.path.join(base_folder, 'A')
    folder_b = os.path.join(base_folder, 'B')
    if not os.path.exists(folder_a) or not os.path.exists(folder_b):
        raise FileNotFoundError(f"Both 'A' and 'B' subfolders must exist in {base_folder}")

    X = []
    y = []

    # Process images in folder A (class 0)
    for filename in os.listdir(folder_a):
        if filename.lower().endswith(('.jpeg', '.jpg')):
            X.append(process_image(os.path.join(folder_a, filename), target_size, normalize))
            y.append(0)

    # Process images in folder B (class 1)
    for filename in os.listdir(folder_b):
        if filename.lower().endswith(('.jpeg', '.jpg')):
            X.append(process_image(os.path.join(folder_b, filename), target_size, normalize))
            y.append(1)

    return np.array(X), np.array(y)
