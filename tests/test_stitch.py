"""
Test stitching functions.

Note: This test was initially generated using Gemini, with manual validation and
editing.
"""

import numpy as np
import pytest
import skimage as sk
from skimage.data import chelsea
from skimage.io import imsave

from oic_toolkit.register import stitch_xy


@pytest.fixture
def mock_tile_dir(tmp_path):
    """
    Generate a temporary series of overlapping tiles using the "chelsea" image.

    This test uses the `tmp_path` fixture to create a temporary directory. pytest will
    remove the directory after testing.

    Yields
    ------
    dict
        A dictionary containing the test parameters and paths:

        - ``dir_path`` (pathlib.Path): The temporary directory for tile storage.
        - ``numX`` (int): The number of horizontal tiles expected.
        - ``numY`` (int): The number of vertical tiles expected.
        - ``overlap`` (float): The percentage of overlap between adjacent tiles.
        - ``expected_tile_shape`` (tuple of int): The expected (height, width)
          dimensions of each individual tile.
    """
    numX, numY = 3, 3
    overlap_percent = 15
    ov_frac = overlap_percent / 100.0

    # Load raw image and convert to grayscale
    img = sk.color.rgb2gray(chelsea())
    h, w = img.shape

    # Calculate tile sizes
    tile_w = int(w / (numX - (numX - 1) * ov_frac))
    tile_h = int(h / (numY - (numY - 1) * ov_frac))

    step_w = int(tile_w * (1 - ov_frac))
    step_h = int(tile_h * (1 - ov_frac))

    idx = 1
    for row in range(numY):
        for col in range(numX):
            y_start = row * step_h
            x_start = col * step_w

            tile = img[y_start : y_start + tile_h, x_start : x_start + tile_w]
            file_path = tmp_path / f"img_{idx:02d}.tif"
            imsave(file_path, sk.img_as_uint(tile))
            idx += 1

    # Yield the parameters to the test function
    yield {
        "dir_path": tmp_path,
        "numX": numX,
        "numY": numY,
        "overlap": overlap_percent,
        "expected_tile_shape": (tile_h, tile_w),
    }


# --- Test Cases ---


def test_stitch_xy_pipeline(mock_tile_dir):
    """Verifies that the stitching pipeline runs and structurally matches the original."""
    data = mock_tile_dir

    # Execute the pipeline
    stitched_image = stitch_xy(
        image_path=data["dir_path"],
        numX=data["numX"],
        numY=data["numY"],
        overlap=data["overlap"],
        file_pattern="img_{ii}.tif",
    )

    # Assertions to verify correctness
    assert stitched_image is not None
    assert isinstance(stitched_image, np.ndarray)
    assert stitched_image.ndim == 2  # Grayscale check

    # Verify canvas dimensions look reasonable (greater than a single tile)
    assert stitched_image.shape[0] > data["expected_tile_shape"][0]
    assert stitched_image.shape[1] > data["expected_tile_shape"][1]

    # Re-load the original ground-truth cat image in grayscale
    original_raw = sk.color.rgb2gray(chelsea())
    original_uint16 = sk.img_as_uint(original_raw)

    assert stitched_image.dtype == original_uint16.dtype  # Datatype preservation check

    # Crop the original image to match the exact dimensions of the stitched canvas.
    # Due to sub-pixel shifts or rounding in the global solver, the canvas size
    # might vary slightly from the absolute raw pixels.
    h_min = min(original_uint16.shape[0], stitched_image.shape[0])
    w_min = min(original_uint16.shape[1], stitched_image.shape[1])

    cropped_original = original_uint16[:h_min, :w_min]
    cropped_stitched = stitched_image[:h_min, :w_min]

    # Compute Structural Similarity Index (SSIM)
    # data_range specifies the dynamic range of the pixel values (65535 for uint16)
    similarity_index = sk.metrics.structural_similarity(
        cropped_original, cropped_stitched, data_range=65535
    )

    print(f"\nStructural Similarity Index (SSIM): {similarity_index:.4f}")

    # A successful stitch with minor alpha-blending at margins should easily score > 0.95
    assert similarity_index > 0.95, (
        f"Stitched image structurally diverged from original. SSIM: {similarity_index}"
    )


def test_stitch_xy_missing_files_raises_error(tmp_path):
    """Ensures that a FileNotFoundError is raised if tile counts don't match assumptions."""
    # Create an empty directory with no files
    with pytest.raises(FileNotFoundError):
        stitch_xy(
            image_path=tmp_path, numX=3, numY=3, overlap=15, file_pattern="img_{ii}.tif"
        )
