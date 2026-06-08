import numpy as np
import skimage as sk
from matplotlib import pyplot as plt


def merge_images(image1, image2, normalize=True):
    """
    Merge images in magenta-green mode. In this mode, images that perfectly overlap will display in gray, but misalignments will show as magenta and green.

    Parameters
    ----------
    image1 : np.array
        Image 1, displayed in magenta
    image2 : np.array
        Image 2, displayed in green
    normalize : bool, optional
        If True, the image range is normalized by the maximum and minimum values in the image, by default True
    """

    if len(image1.shape) == 3:
        image1 = sk.color.rgb2gray(image1)

    if len(image2.shape) == 3:
        image2 = sk.color.rgb2gray(image2)

    # Trim images to the smaller of the two dimensions
    hf = np.min([image1.shape[0], image2.shape[0]])
    wf = np.min([image1.shape[1], image2.shape[1]])

    image1 = image1[:hf, :wf]
    image2 = image2[:hf, :wf]
    
    if not (image1.shape == image2.shape):
        raise ValueError(f"The two images are not the same shape. (Image1:{image1.shape}, Image2:{image2.shape})")

    if normalize:
        # Normalize images to make sure they look good
        image1 = sk.exposure.rescale_intensity(image1,
                                               in_range="image",
                                               out_range=(0.0, 1.0))
        

        image2 = sk.exposure.rescale_intensity(image2, 
                                               in_range="image",
                                               out_range=(0.0, 1.0))
        
    image1 = sk.util.img_as_ubyte(image1)
    image2 = sk.util.img_as_ubyte(image2)

    merged = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)

    merged[..., 0] = image1
    merged[..., 1] = image2
    merged[..., 2] = image1

    return merged
