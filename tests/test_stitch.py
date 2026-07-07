import numpy as np
import skimage

from oic_toolkit import register

final = register.stitch_xy(
    r"D:\Projects\henderson-lab-brain-gvb\processed\20260707_AM1c-s11-r002_Plate_4555",
    7,
    9,
)

skimage.io.imsave("test.png", final.astype(np.uint8))
