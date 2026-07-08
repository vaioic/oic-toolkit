"""
============
2D stitching
============

This example uses the classic scikit-image cat Chelsea to demonstrate 2D stitching. The
image is divided into a grid of 9 tiles, with 15 percent overlap.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import skimage as sk
from skimage.data import chelsea
from skimage.io import imsave

# Import your active processing functions
from oic_toolkit.register import stitch_xy

# Setup a temporary directory path inside the example execution runtime
mock_dir = Path("./temp_cat_tiles")
mock_dir.mkdir(exist_ok=True)

# 1. Slice up the image to generate our synthetic test data
img = sk.color.rgb2gray(chelsea())
h, w = img.shape
numX, numY, overlap_percent = 3, 3, 15
ov_frac = overlap_percent / 100.0

tile_w = int(w / (numX - (numX - 1) * ov_frac))
tile_h = int(h / (numY - (numY - 1) * ov_frac))
step_w = int(tile_w * (1 - ov_frac))
step_h = int(tile_h * (1 - ov_frac))

idx = 1
for row in range(numY):
    for col in range(numX):
        y_start, x_start = row * step_h, col * step_w
        tile = img[y_start : y_start + tile_h, x_start : x_start + tile_w]
        imsave(mock_dir / f"img_{idx:02d}.tif", sk.img_as_uint(tile))
        idx += 1

# #############################################################################
# Running the Pipeline Execution
# ------------------------------
# Now we call our core `stitch_xy` function.

stitched_result = stitch_xy(
    image_path=mock_dir, numX=numX, numY=numY, overlap=overlap_percent
)

# #############################################################################
# Visualizing the Blended Result
# ------------------------------
# Let's render the output directly into our documentation matrix page.

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(stitched_result, cmap="gray")
ax.set_title(
    f"Final Stitched Canvas Canvas ({stitched_result.shape[1]}x{stitched_result.shape[0]})"
)
ax.axis("off")
plt.show()

# Clean up temporary disk files when compilation finishes
for f in mock_dir.glob("*.tif"):
    f.unlink()
mock_dir.rmdir()
