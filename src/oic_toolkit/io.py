"""Functions related to reading and writing files."""

import tifffile

from oic_toolkit._internal import validate_path


def export_pyramid_tiff(output_path, image):
    """
    Export QuPath/Fiji-compatible pyramid tiffs.

    Pyramid TIFFs have several resolutions which help with file loading speeds. This
    function writes TIFFs which should be able to be opened using QuPath and Fiji with
    the Bio-Formats plugin.

    Currently, the function downsamples images by 2, 4, and 8.

    Parameters
    ----------
    output_path : str or Path
        Path to write to
    image : ndarray
        Image data in (C, Y, X) order
    """
    output_path = validate_path(output_path)
    print(output_path)

    with tifffile.TiffWriter(output_path, bigtiff=True) as tiff:
        print("Writing TIFF")
        options = dict(tile=(256, 256), compression="jpeg")

        # Write the base resolution (Level 0)
        tiff.write(
            image,
            subifds=3,  # Number of downsampled levels to follow
            metadata={"axes": "CYX"},
            **options,
        )

        # Write downsampled levels
        for factor in [2, 4, 8]:
            # Downsample data using simple indexing
            downsampled = image[:, ::factor, ::factor]
            tiff.write(downsampled, subfiletype=1, **options)
