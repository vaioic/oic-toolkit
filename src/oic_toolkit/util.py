from pathlib import Path

import numpy as np
import skimage as sk


def downsample_image(image, output_path=None, ds_factor=8, normalize=False):

    image_ds = image[::ds_factor, ::ds_factor]

    if normalize:
        image_ds = sk.exposure.rescale_intensity(image_ds, out_range=(0.0, 1.0))

    if output_path:

        if isinstance(output_path, str):
            output_path = Path(output_path)

        if not output_path.suffix:
            raise ValueError(f"The input path is not a valid filename")
        
        if not (output_path.parent).exists():
            (output_path.parent).mkdir(parents=True)
        
        sk.io.imsave(output_path, image_ds)

        print(f"Downsampled image saved to {str(output_path)}.")
    else:
        return image_ds

def resize_from_center(image, target_shape):
    """
    Symmetrically pads or crops an image relative to its center.
    
    Parameters:
        image (ndarray): Input image array (2D grayscale or 3D color).
        target_shape (tuple): Desired (height, width) of the output.
    """
    # Handle color channels by isolating spatial dimensions
    img_h, img_w = image.shape[:2]
    tgt_h, tgt_w = target_shape[:2]
    
    # --- STEP 1: VERTICAL AXIS (HEIGHT) ---
    if tgt_h > img_h:
        # Grow height: calculate top and bottom padding
        pad_top = (tgt_h - img_h) // 2
        pad_bottom = tgt_h - img_h - pad_top
        crop_top, crop_bottom = 0, img_h
    else:
        # Shrink height: calculate top and bottom crop indices
        crop_top = (img_h - tgt_h) // 2
        crop_bottom = crop_top + tgt_h
        pad_top, pad_bottom = 0, 0

    # --- STEP 2: HORIZONTAL AXIS (WIDTH) ---
    if tgt_w > img_w:
        # Grow width: calculate left and right padding
        pad_left = (tgt_w - img_w) // 2
        pad_right = tgt_w - img_w - pad_left
        crop_left, crop_right = 0, img_w
    else:
        # Shrink width: calculate left and right crop indices
        crop_left = (img_w - tgt_w) // 2
        crop_right = crop_left + tgt_w
        pad_left, pad_right = 0, 0

    # --- STEP 3: APPLY CROP & PAD ---
    # First, slice the image down if shrinking was required
    output = image[crop_top:crop_bottom, crop_left:crop_right]
    
    # Second, pad the image out if growing was required
    if tgt_h > img_h or tgt_w > img_w:
        # Define padding for height and width axes
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        
        # Append an empty padding tuple if the image has color channels
        if image.ndim == 3:
            pad_width += ((0, 0),)
            
        # constant_values=0 fills the background with black
        output = np.pad(output, pad_width=pad_width, mode='constant', constant_values=0)
        
    return output
