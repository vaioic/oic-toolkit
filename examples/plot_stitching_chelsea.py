"""
====================
2D stitching
====================

This example uses the classic scikit-image cat Chelsea to demonstrate 2D stitching. The
image is divided into a grid of 9 tiles, with 15 percent overlap.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import skimage as sk
from skimage.data import chelsea
from skimage.io import imsave

# Import your active processing functions
from oic_toolkit.register import generate_tiled_image, stitch_xy

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

abs_x, abs_y = stitch_xy(
    image_path=mock_dir, numX=numX, numY=numY, overlap=overlap_percent
)

stitched_result = generate_tiled_image(mock_dir, abs_x=abs_x, abs_y=abs_y)

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

###############################
# Saving the stitching coordinates
import pandas as pd

tile_indices = list(range(len(abs_x)))

# Build the DataFrame and export to CSV
df = pd.DataFrame({"tile_index": tile_indices, "abs_x": abs_x, "abs_y": abs_y})
df.to_csv(mock_dir / "tile_positions.csv", index=False)

####


df = pd.read_csv(mock_dir / "tile_positions.csv")

# Extract the columns directly back into simple Python lists
abs_x = df["abs_x"].tolist()
abs_y = df["abs_y"].tolist()

stitched_result2 = generate_tiled_image(mock_dir, abs_x=abs_x, abs_y=abs_y)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(stitched_result2, cmap="gray")
ax.set_title(
    f"Stitched Image ({stitched_result2.shape[1]}x{stitched_result2.shape[0]})"
)
ax.axis("off")
plt.show()

# Clean up temporary disk files when compilation finishes
for f in mock_dir.glob("*.tif"):
    f.unlink()
for f in mock_dir.glob("*.csv"):
    f.unlink()
mock_dir.rmdir()
