# data wrangling
import numpy as np
# image processing
import tifffile
from PIL import Image
from pathlib import Path
import shutil

def convert_tiff_channel_to_jpg(tiff_path, channel_index=0, output_size=(24, 24), create_negative=True):
    """
    Convert a specific channel from a TIFF file to a JPG, with option to create negative.

    Parameters:
    -----------
    tiff_path : str or Path
        Path to the input TIFF file
    channel_index : int
        Index of the channel to process (default: 0)
    output_size : tuple
        Size of the output JPG (width, height) (default: (24, 24))
    create_negative : bool
        Whether to create a negative of the image (default: True)

    Returns:
    --------
    str
        Path to the saved JPG file
    """
    # Convert path to Path object
    tiff_path = Path(tiff_path)

    # Read the TIFF file
    image = tifffile.imread(str(tiff_path))

    # Get the specified channel (now using the last dimension)
    if image.shape[2] == 4:
        channel = image[:, :, channel_index]
    else:
        channel = image[channel_index]

    # Create negative if requested
    if create_negative:
        if channel.dtype == np.uint16:
            processed = 65535 - channel
            processed = (processed / 256).astype(np.uint8)
        elif channel.dtype == np.uint8:
            processed = 255 - channel
        else:
            processed = ((channel - channel.min()) /
                        (channel.max() - channel.min()) * 255).astype(np.uint8)
            processed = 255 - processed
    else:
        # Just normalize to 8-bit if not creating negative
        if channel.dtype == np.uint16:
            processed = (channel / 256).astype(np.uint8)
        elif channel.dtype == np.uint8:
            processed = channel
        else:
            processed = ((channel - channel.min()) /
                        (channel.max() - channel.min()) * 255).astype(np.uint8)

    # Convert to PIL Image and resize
    pil_image = Image.fromarray(processed)
    resized_image = pil_image.resize(output_size, Image.Resampling.LANCZOS)

    # Create output filename
    output_path = tiff_path.parent / f"{tiff_path.stem}_channel{channel_index}.jpg"

    # Save the image
    resized_image.save(output_path, quality=95)

    return str(output_path)

# Now, let's create a function to process all TIFF files in a folder
def process_folder_tiffs(folder_path, dest_folder, channel_index=0, output_size=(224, 224), create_negative=True):
    """
    Process all TIFF files in a folder and convert specified channels to JPG.

    Parameters:
    -----------
    folder_path : str or Path
        Path to the folder containing TIFF files
    dest_folder : str or Path
        Path to the destination folder for JPG files
    channel_index : int
        Index of the channel to process (default: 0)
    output_size : tuple
        Size of output JPG (width, height) (default: (224, 224))
    create_negative : bool
        Whether to create negative of the image (default: True)

    Returns:
    --------
    dict
        Summary of processing results
    """
    folder_path = Path(folder_path)
    dest_folder = Path(dest_folder)

    # Ensure destination folder exists
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Initialize results dictionary
    results = {
        'processed_files': [],
        'errors': [],
        'total_processed': 0
    }

    # Get all TIFF files in the folder
    tiff_files = list(folder_path.glob('*.tif')) + list(folder_path.glob('*.tiff'))
    total_files = len(tiff_files)

    # Process each TIFF file
    for i, tiff_file in enumerate(tiff_files, 1):
        try:
            # Process the file
            output_path = convert_tiff_channel_to_jpg(
                tiff_file,
                channel_index=channel_index,
                output_size=output_size,
                create_negative=create_negative
            )

            # Move the processed file to the destination folder
            dest_file = dest_folder / output_path

            try:
                shutil.move(output_path, dest_folder)
            except Exception as e:
                error_msg = f"{tiff_file} is already processed! {str(e)}"
                print(f"{error_msg}")

            results['processed_files'].append(dest_file)
            results['total_processed'] += 1

            # Print progress
            print(f"Processed {i}/{total_files}: {tiff_file}")

        except Exception as e:
            error_msg = f"Error processing {tiff_file}: {str(e)}"
            results['errors'].append(error_msg)
            print(f"Error: {error_msg}")

    return results
