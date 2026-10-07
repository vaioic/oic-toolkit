"""Miscellaneous functions for image processing."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import skimage as sk
from bioio import BioImage
from tqdm import tqdm


def downsample_image(image, output_path=None, ds_factor=8, normalize=False):
    """
    Downsample and optionally save an image.

    Parameters
    ----------
    image : ndarray
        Input image
    output_path : str or Path, optional
        Output path to save downsampled image, by default None. If None, the output
        image will be returned as a variable instead.
    ds_factor : int, optional
        Downsample factor, by default 8
    normalize : bool, optional
        If True, rescale image intensity to be between 0.0 - 1.0, by default False

    Returns
    -------
    image_ds : ndarray, optional
        Downsampled image

    Raises
    ------
    ValueError
        Error occurs if the input path is not a valid filename (e.g., it is a directory)
    """
    image_ds = image[::ds_factor, ::ds_factor]

    if normalize:
        image_ds = sk.exposure.rescale_intensity(image_ds, out_range=(0.0, 1.0))

    if output_path:
        if isinstance(output_path, str):
            output_path = Path(output_path)

        if not output_path.suffix:
            raise ValueError("The input path is not a valid filename")

        if not (output_path.parent).exists():
            (output_path.parent).mkdir(parents=True)

        sk.io.imsave(output_path, image_ds)

        print(f"Downsampled image saved to {str(output_path)}.")
    else:
        return image_ds


def resize_from_center(image, target_shape):
    """
    Symmetrically pad or crop an image around the center.

    Parameters
    ----------
    image : ndarray
        Input image
    target_shape : list-like
        Specify the output shape as (height, width)

    Returns
    -------
    output : ndarray
        Resized image
    """
    img_h, img_w = image.shape[:2]
    tgt_h, tgt_w = target_shape[:2]

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

    output = image[crop_top:crop_bottom, crop_left:crop_right]

    if tgt_h > img_h or tgt_w > img_w:
        # Define padding for height and width axes
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))

        # Append an empty padding tuple if the image has color channels
        if image.ndim == 3:
            pad_width += ((0, 0),)

        output = np.pad(output, pad_width=pad_width, mode="constant", constant_values=0)

    return output


def get_file_metadata(fp):
    """
    Extract metadata from microscopy datasets.

    Note: This was only tested on ND2 files.

    Parameters
    ----------
    fp : str or Path
        Path to the image file

    Returns
    -------
    dict
        A dictionary containing standard metadata information such as image dimensions,
        laser power, and objective information.
    """
    img = BioImage(fp)

    # Get the "standard" metadata
    meta_standard = img.standard_metadata
    # print(f"{(meta_standard.pixel_size_x):.3f}")
    meta_dict = img.metadata.model_dump()

    map_annots = meta_dict.get("structured_annotations", {}).get("map_annotations", [])

    for entry in map_annots:
        # 'value' is the dict containing Nikon chunks (e.g., ImageMetadataLV)
        chunks = entry.get("value", {})

        for chunk_name, chunk_str in chunks.items():
            if chunk_name == "ImageMetadataSeqLV|0":
                parsed_content = json.loads(chunk_str)

                properties = (
                    parsed_content.get("SLxPictureMetadata", {})
                    .get("PicturePlanes", {})
                    .get("SampleSetting", {})
                    .get("0", {})
                )

                md_properties = properties.get("CameraSetting", {}).get(
                    "PropertiesQuality", {}
                )

                md_objectives = properties.get("ObjectiveSetting", {})

                # Note: Similar metadata exists in the SLVtextmetadata - I think Nikon uses that in the image properties display.

                # This string is not formatted correctly in the JSON, so read as string
                md_scanner = properties.get("SpecSettings", "")
                scan_speed = re.search(
                    r"{Scan Speed}:\s*([\d.]+)", md_scanner, re.DOTALL
                )
                pinhole_size = re.search(
                    r"{Pinhole Size\(um\)}:\s*([\d.]+)", md_scanner, re.DOTALL
                )
                scanner_zoom = re.search(
                    r"{Scanner Zoom}:\s*([\d.]+)", md_scanner, re.DOTALL
                )
                objzoom = parsed_content.get("SLxPictureMetadata", {}).get("Zoom", "")

                # print(scanner_zoom.group(1))

                # Store data as dict
                file_metadata = {
                    "Filename": fp.name,
                    "Image_Dimensions": meta_standard.dimensions_present,
                    "Image_Size_C": meta_standard.image_size_c,
                    "Image_Size_X": meta_standard.image_size_x,
                    "Image_Size_Y": meta_standard.image_size_y,
                    "Image_Size_Z": meta_standard.image_size_z,
                    "Pixel_Size": f"{(meta_standard.pixel_size_x):.2f}",
                    "Objective": md_objectives["ObjectiveName"],
                    "ObjectiveMag": md_objectives["ObjectiveMag"],
                    "ObjectiveNA": md_objectives["ObjectiveNA"],
                    "Scan_speed": scan_speed.group(1),
                    "Pinhole_size_um": pinhole_size.group(1),
                    "Scanner_zoom": scanner_zoom.group(1),
                    "Image zoom": objzoom,
                    "Ch1_Name": md_properties["CH1ChannelDyeName"],
                    "Ch1_LaserPower": md_properties["CH1LaserPower"],
                    "Ch1_PMTVoltage": md_properties["CH1PMTHighVoltage"],
                    "Ch1_PMTOffset": md_properties["CH1PMTOffset"],
                    "Ch2_Name": md_properties["CH2ChannelDyeName"],
                    "Ch2_LaserPower": md_properties["CH2LaserPower"],
                    "Ch2_PMTVoltage": md_properties["CH2PMTHighVoltage"],
                    "Ch2_PMTOffset": md_properties["CH2PMTOffset"],
                    "Ch3_Name": md_properties["CH3ChannelDyeName"],
                    "Ch3_LaserPower": md_properties["CH3LaserPower"],
                    "Ch3_PMTVoltage": md_properties["CH3PMTHighVoltage"],
                    "Ch3_PMTOffset": md_properties["CH3PMTOffset"],
                    "Ch4_Name": md_properties["CH4ChannelDyeName"],
                    "Ch4_LaserPower": md_properties["CH4LaserPower"],
                    "Ch4_PMTVoltage": md_properties["CH4PMTHighVoltage"],
                    "Ch4_PMTOffset": md_properties["CH4PMTOffset"],
                }

    return file_metadata


def write_metadata_dump(fp, output_dir):
    """
    Write metadata from image file to text file.

    Note: Only tested on ND2 files

    Parameters
    ----------
    fp : str or Path
        Path to image file
    output_dir : str or Path
        Path to output directory
    """
    img = BioImage(fp)

    # Get the "standard" metadata
    meta_standard = img.standard_metadata
    # print(f"{(meta_standard.pixel_size_x):.3f}")
    meta_dict = img.metadata.model_dump()

    map_annots = meta_dict.get("structured_annotations", {}).get("map_annotations", [])

    with open(output_dir / "dump.txt", "w") as f:
        for entry in map_annots:
            # 'value' is the dict containing Nikon chunks (e.g., ImageMetadataLV)
            chunks = entry.get("value", {})

        for chunk_name, chunk_str in chunks.items():
            f.write(f"\n--- Section: {chunk_name} ---\n")

            try:
                # Attempt to parse and write pretty JSON
                parsed_content = json.loads(chunk_str)
                f.write(json.dumps(parsed_content, indent=4))
            except (json.JSONDecodeError, TypeError):
                # Fallback: write as raw text if not valid JSON
                f.write(str(chunk_str))

            f.write("\n")


def get_metadata_all_files(image_dir, output_dir):
    """
    Extract metadata from image files in a directory.

    Parameters
    ----------
    image_dir : str or Path
        Path to folder containing image files.
    output_dir : str or Path
        Path to output folder

    Raises
    ------
    ValueError
        Input directory path was not a str or Path.
    ValueError
        Output directory was not a str or Path.
    """
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)
    elif isinstance(image_dir, Path):
        pass
    else:
        raise ValueError("Expected input path to be a str or Path.")

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    elif isinstance(output_dir, Path):
        pass
    else:
        raise ValueError("Expected output path to be a str or Path.")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    file_list = image_dir.glob("*.nd2")

    all_metadata = []

    for f in tqdm(file_list):
        md = get_file_metadata(f)
        all_metadata.append(md)

    df = pd.DataFrame(all_metadata)
    df.to_csv(output_dir / "all_metadata.csv")
