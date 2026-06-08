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

