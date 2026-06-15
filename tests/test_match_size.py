import numpy as np
import skimage as sk
from matplotlib import pyplot as plt

from oic_toolkit import display, register

target = sk.data.cat()
target_gray = sk.color.rgb2gray(target)

moving_smaller = sk.transform.rescale(target, 0.50, channel_axis=-1)

# moving_smaller = sk.transform.rescale(target, 0.50, preserve_range=True)

moving_matched = register.match_size(target_gray, moving_smaller)

print(f"Target shape: {target.shape}.")
print(f"Rescaled moving shape: {moving_smaller.shape}")      
print(f"Moving matched shape: {moving_matched.shape}")

# merge = display.merge_images(target, moving_matched)
plt.imshow(moving_matched)
plt.show()

moving_larger = sk.transform.rescale(target, 1.50, channel_axis=-1)

# moving_larger = sk.transform.rescale(target, 1.50, preserve_range=True)

print(f"Rescaled moving shape: {moving_larger.shape}")      

moving_matched = register.match_size(target_gray, moving_larger)


print(f"Moving matched shape: {moving_matched.shape}")

# merge = display.merge_images(target, moving_matched)
plt.imshow(moving_matched)
plt.show()
