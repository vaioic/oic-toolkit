"""
====================================================
Merge images
====================================================

In this example, we use the Chelsea cat image to demonstrate how the ``merge_images``
function works. The image will be shifted slightly to demonstrate how this works.
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from scipy import ndimage
from skimage.data import chelsea

import oic_toolkit

# Import the Chelsea image as the example
target_img = sk.color.rgb2gray(chelsea())
h, w = target_img.shape

# Shift the image
true_shift = (-18.5, 25.2)

# Create the moved image
moving_img = ndimage.shift(target_img, shift=(-true_shift[0], -true_shift[1]), cval=0.0)

# Display the two images to show the translational shift
merged = oic_toolkit.display.merge_images(target_img, moving_img)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(merged)
ax.axis("off")
plt.tight_layout()
plt.show()
