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



