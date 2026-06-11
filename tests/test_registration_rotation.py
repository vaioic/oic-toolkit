import skimage as sk
from matplotlib import pyplot as plt

from oic_toolkit import display, register

original_image = sk.data.cat()

print(original_image.shape)

original_image = sk.color.rgb2gray(original_image)

# Rotate and scale the image

rotation_degrees = 25
scaling = 1.15

modified_image = sk.transform.rotate(original_image, rotation_degrees) #TODO: Test with a non-center origin
modified_image = sk.transform.rescale(modified_image, scaling)

print(f"Modified image shape: {modified_image.shape}")

target, moving = register.pad_images(original_image, modified_image)

# Try the registration

shift_rotation, shift_scale, corrected = register.rotation_scale_phasecorr(target, moving)

print(f"Recovered rotation: {shift_rotation}. Expected: {rotation_degrees}")
print(f"Recovered scale: {shift_scale}. Expected: {scaling}")


# Plot the results

fig = plt.figure()

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)

ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)

ax1.imshow(target)
ax2.imshow(moving)

merge_or = display.merge_images(target, moving)

ax3.imshow(merge_or)

ax4.imshow(corrected)

merge_corr = display.merge_images(target, corrected)
ax5.imshow(merge_corr)


plt.show()
plt.close()
