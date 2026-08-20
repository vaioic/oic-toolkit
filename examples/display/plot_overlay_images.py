"""
====================================================
Overlay masks over images
====================================================

This example shows how to use the ``overlay_mask`` function to generate overlay images
to validate segmentation masks.
"""

import skimage
from matplotlib import pyplot as plt

import oic_toolkit

# %%
# Import example images and a mask for testing
# --------------------------------------------

data = skimage.data.cell()

thresh = skimage.filters.threshold_otsu(data)
mask = data > thresh

# %%
# Generate the overlay using default parameters
# ---------------------------------------------
overlay = oic_toolkit.display.overlay_mask(data, mask)

plt.imshow(overlay)
plt.show()

# %%
# Generate a different overlay, changing the mask color and the transparency
# --------------------------------------------------------------------------
#
# Both the color and transparency of the displayed mask overlay can be adjusted using
# the ``mask_color`` and ``alpha`` properties.
#
# ``mask_color`` must be a normalized (i.e., between 0 and 1) RGB vector
# ``alpha`` must be between 0 and 1. A higher value will mean that the mask will appear
# more opaque.

overlay = oic_toolkit.display.overlay_mask(data, mask, mask_color=(1, 0, 1), alpha=0.8)

plt.imshow(overlay)
plt.show()
