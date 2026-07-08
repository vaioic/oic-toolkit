"""Test functions in io.py."""

import numpy as np
import tifffile

from oic_toolkit import io


def test_export_pyramid_tiff_runs(tmp_path):
    """Test pyramid file export."""
    # Note: Opening this file in QuPath, the Image tab should show values in the
    # Pyramid section
    mock_data = np.random.randint(0, 255, (3, 20480, 20480), dtype=np.uint8)

    # Write a test file
    output_path = tmp_path / "test_pyramid.ome.tiff"

    io.export_pyramid_tiff(output_path, mock_data)

    assert tmp_path.exists()

    with tifffile.TiffFile(output_path) as tif:
        # Check that it is recognized as an OME-TIFF
        assert tif.is_ome

        # Access the main series (the image data chain)
        series = tif.series[0]

        # Verify that sub-resolutions (levels) exist
        # If your function creates 3 levels, levels length should be 3
        assert len(series.levels) > 1

        # Verify the dimensions of each level decrease appropriately
        assert series.levels[0].shape == (3, 20480, 20480)
        assert series.levels[1].shape == (3, 20480 // 2, 20480 // 2)
        assert series.levels[2].shape == (3, 20480 // 4, 20480 // 4)
        assert series.levels[3].shape == (3, 20480 // 8, 20480 // 8)
