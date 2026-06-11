import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
from scipy.fft import fft2, fftshift

from oic_toolkit import display, register, util

original_image = sk.data.cat()

print(f"Original image shape: {original_image.shape}")

original_image = sk.color.rgb2gray(original_image)

# register.plot_diff_gauss(original_image)

# Rotate and scale the image

rotation_degrees = 25
scaling = 1.15
translation = (15, 21)

modified_image = sk.transform.rotate(original_image, rotation_degrees) #TODO: Test with a non-center origin
modified_image = sk.transform.rescale(modified_image, scaling)

# Also try translation
tform_translate = sk.transform.SimilarityTransform(translation=translation)
modified_image = sk.transform.warp(modified_image, tform_translate)

print(f"Modified image shape: {modified_image.shape}")

target, moving = register.pad_images(original_image, modified_image)

# Calculate the padding dimensions to adjust the Hanning window
h0, w0 = original_image.shape
hp, wp = modified_image.shape

if hp > h0:
    h_window = h0 + ((hp - h0) // 2)

if wp > w0:
    w_window = w0 + ((wp - w0) // 2)

print(f"Window size: {(h_window, w_window)}")

shift_rotation, shift_scale, shift_translation, corrected = register.log_polar_phasecorr(target, moving, sigma_low=5, sigma_high=30, window_size=(h_window, w_window))

print(f"Recovered rotation: {shift_rotation}. Expected: {rotation_degrees}")
print(f"Recovered scale: {shift_scale}. Expected: {scaling}")
print(f"Recovered translation: {shift_translation}. Expected: {translation}")


# Plot the results

fig = plt.figure()

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)

ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

ax1.imshow(target)
ax1.set_title("Original (padded)")

ax2.imshow(moving)
ax2.set_title("Moved (padded)")

merge_or = display.merge_images(target, moving)

ax3.imshow(merge_or)
ax3.set_title("Merged (original)")

ax4.imshow(corrected)
ax4.set_title("Corrected image")

# Add a fudge to see if I can correct it


merge_corr = display.merge_images(target, corrected)
ax5.imshow(merge_corr)
ax5.set_title("Merged (corrected)")

# Add the optical flow to see if it can fix the final step
target_cropped = util.resize_from_center(target, original_image.shape)
corrected_cropped = util.resize_from_center(corrected, original_image.shape)

v, u = sk.registration.optical_flow_tvl1(target_cropped, corrected_cropped, prefilter=True)

nr, nc = target_cropped.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

corrected_twice = sk.transform.warp(corrected_cropped, np.array([row_coords + v, col_coords + u]), mode='edge')

merge_twice = display.merge_images(target_cropped, corrected_twice)

ax6.imshow(merge_twice)

plt.show()
plt.close()
