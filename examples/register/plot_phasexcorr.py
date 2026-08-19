"""
====================================================
Registering two images using phase cross-correlation
====================================================

This example uses the classic scikit-image cat Chelsea to demonstrate image registration
using phase cross-correlation. It is important to note that this approach only works for
__translational__ shifts and will not work for rotational or scaling differences.
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
from scipy import ndimage
from skimage.data import chelsea

import oic_toolkit

# %%
# Example images
# --------------

# Import the Chelsea image as the example
target_img = sk.color.rgb2gray(chelsea())
h, w = target_img.shape

# Define the actual shift
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

# %%
# Correct the images
# ------------------

# Perform the cross-correlation, returning the corrected images
results, target_crop, moving_crop = oic_toolkit.register.phasexcorr(
    target_img, moving_img, return_corrected=True
)

print(f"Calculated Shift (y, x): {results['shift']}. Expected: {true_shift}.")
print(f"Target Cropped Shape:    {target_crop.shape}")
print(f"Moving Cropped Shape:    {moving_crop.shape}")

# Visualize the resulting images and the difference between the target and corrected images
fig, axes = plt.subplots(2, 3, figsize=(12, 4))

axes[0, 0].imshow(target_img, cmap="gray")
axes[0, 0].set_title("Original (reference) image")

axes[0, 1].imshow(moving_img, cmap="gray")
axes[0, 1].set_title("Unregistered moving Image")

axes[1, 0].imshow(target_crop, cmap="gray")
axes[1, 0].set_title("Cropped reference image")

axes[1, 1].imshow(moving_crop, cmap="gray")
axes[1, 1].set_title("Cropped and registered moving Image")

# Absolute difference of cropped overlap to verify alignment
diff = np.abs(target_crop - moving_crop)
axes[1, 2].imshow(diff, cmap="magma")
axes[1, 2].set_title("Absolute Diff (Cropped Overlap)")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
