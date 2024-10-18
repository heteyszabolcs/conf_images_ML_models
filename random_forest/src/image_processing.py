# data wrangling
import numpy as np
# image processing
import tifffile
from PIL import Image
from pathlib import Path

def process_image(file_path, target_size, normalize):
    """
    Process a single image file.

    Args:
    file_path (str): Path to the image file.
    target_size (tuple): Desired size of the output image (height, width).
    normalize (bool): If True, normalize pixel values to [0, 1].

    Returns:
    numpy.ndarray: Flattened image array.
    """
    with Image.open(file_path) as img:
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize the image
        img = img.resize(target_size)

        # Convert to numpy array
        img_array = np.array(img)

        # Normalize if required
        if normalize:
            img_array = img_array.astype(np.float32) / 255.0

        # Flatten the image
        return img_array.reshape(-1)
