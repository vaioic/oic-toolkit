import numpy as np
import skimage as sk


def difference_of_gaussians(image, d_min=3, d_max=10):

    # Calculate the approximate sigma values from the diameters
    sigma_low = d_min / (2 * (2 ** 0.5))
    sigma_high = d_max / (2 * (2 ** 0.5))

    # Note: The filter currently normalizes the image intensity
    g_low = sk.filters.gaussian(image, sigma=sigma_low)
    g_high = sk.filters.gaussian(image, sigma=sigma_high)

    dog_image = g_low - g_high
    dog_normalized = (dog_image - np.min(dog_image))/(np.max(dog_image) - np.min(dog_image))

    return dog_normalized


def match_color(image, target_rgb, radius=10):
    """
    Segment objects by color

    This function works by first converting the image into LAB color space. The matching colors are then defined by a circle centered around the target color.

    Parameters
    ----------
    image : ndarray
        Image to segment, must be RGB
    target_rgb : tuple
        Target color to match (R, G, B)
    radius : int, optional
        Radius of matching color, by default 10. The larger the radius, the more colors will be matched

    Returns
    -------
    mask : ndaarray
        Binary mask of pixels that match the color
    """

    if len(image.shape) < 3:
        raise ValueError("Expected the input image to be RGB.")
    elif len(image.shape) > 4:
        raise ValueError(f"Invalid image shape {image.shape}. Expected a 3D array with format (height, width, channels) where color must be 3 (RGB) or 4 (RGBA).")        
    else:
        image = image[..., :3]
        
    # Convert image to LAB
    image_LAB = sk.color.rgb2lab(image)

    # Convert the target color
    target_rgb = np.array(target_rgb, dtype=np.float32) / 255.0
    target_rgb = target_rgb.reshape(1, 1, 3)
    
    target_LAB = sk.color.rgb2lab(target_rgb)

    # Calculate color radius
    mask = ((image_LAB[..., 1] - target_LAB[0,0,1]) ** 2 + (image_LAB[..., 2] - target_LAB[0,0,2]) ** 2) <= (radius ** 2)

    return mask
