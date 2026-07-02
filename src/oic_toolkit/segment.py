import numpy as np
import skimage as sk
from scipy import ndimage as ndi


def difference_of_gaussians(image, d_min=3.0, d_max=10.0):
    """
    Calculate the difference of Gaussians image.

    The difference of Gaussians can be used to find features of a specific size such as
    lines and spots.

    Parameters
    ----------
    image : ndarray
        Input image
    d_min : float, optional
        Minimum object or feature size in pixels, by default 3.0
    d_max : float, optional
        Maximum object or feature size in pixels, by default 10.0

    Returns
    -------
    dog_normalized : ndarray
        Difference of Gaussians image, normalized to 0.0 - 1.0
    """
    # Calculate the approximate sigma values from the diameters
    sigma_low = d_min / (2 * (2**0.5))
    sigma_high = d_max / (2 * (2**0.5))

    # Note: The filter currently normalizes the image intensity
    g_low = sk.filters.gaussian(image, sigma=sigma_low)
    g_high = sk.filters.gaussian(image, sigma=sigma_high)

    dog_image = g_low - g_high
    dog_normalized = (dog_image - np.min(dog_image)) / (
        np.max(dog_image) - np.min(dog_image)
    )

    return dog_normalized


def match_color(image, target_rgb, radius=10):
    """
    Segment objects by color.

    This function works by first converting the image into LAB color space. The matching
    colors are then defined by a circle centered around the target color.

    Parameters
    ----------
    image : ndarray
        Image to segment, must be RGB
    target_rgb : tuple
        Target color to match (R, G, B)
    radius : int, optional
        Radius of matching color, by default 10. The larger the radius, the more colors
        will be matched

    Returns
    -------
    mask : ndarray
        Binary mask of pixels that match the color
    """
    if len(image.shape) < 3:
        raise ValueError("Expected the input image to be RGB.")
    elif len(image.shape) > 4:
        raise ValueError(
            f"Invalid image shape {image.shape}. Expected a 3D array with format (height, width, channels) where color must be 3 (RGB) or 4 (RGBA)."
        )
    else:
        image = image[..., :3]

    # Convert image to LAB
    image_LAB = sk.color.rgb2lab(image)

    # Convert the target color
    target_rgb = np.array(target_rgb, dtype=np.float32) / 255.0
    target_rgb = target_rgb.reshape(1, 1, 3)

    target_LAB = sk.color.rgb2lab(target_rgb)

    # Calculate color radius
    mask = (
        (image_LAB[..., 1] - target_LAB[0, 0, 1]) ** 2
        + (image_LAB[..., 2] - target_LAB[0, 0, 2]) ** 2
    ) <= (radius**2)

    return mask


def separate_objects(mask, h_value=3.0, min_distance=10, threshold_abs=5):
    """
    Run the watershed algorithm to separate objects.

    This function uses a distance-transform to carry out the watershed operation. This
    works best for circular objects.

    Parameters
    ----------
    mask : ndarray
        Binary mask separating the foreground (True) from the background (False)
    h_value : float, optional
        Suppress maxima less than this value. Increase this value if the resulting
        masks are over-segmented. By default, 3.0.
    min_distance = int, optional
        Peaks (objects) should be at least this distance apart. By default, 10.

    Returns
    -------
    labels : ndarray
        Label matrix of individual objects. Each unique object will be labeled
        sequentially with the same ID.
    """
    # Calculate distance transform
    distance = ndi.distance_transform_edt(mask)

    # Suppress maxima to avoid over-segmentation
    h_suppressed = sk.morphology.h_maxima(distance, h_value)
    filtered_distance = np.where(h_suppressed, distance, 0)

    coords = sk.feature.peak_local_max(
        filtered_distance, footprint=None, min_distance=10, threshold_abs=5
    )

    peak_mask = np.zeros(distance.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    # h = 3.0  # Adjust this value like MATLAB's peak prominence
    # h_max = sk.morphology.h_maxima(distance, h)
    # markers, _ = ndi.label(h_max)

    labels = sk.segmentation.watershed(-distance, markers, mask=mask)

    return labels
