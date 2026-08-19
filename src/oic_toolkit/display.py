"""Functions for plotting or generating images."""

import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector


def merge_images(image1, image2, normalize=True):
    """
    Merge images in magenta-green mode.

    In this mode, images that perfectly
    overlap will display in gray, but misalignments will show as magenta and
    green.

    Parameters
    ----------
    image1 : np.array
        Image 1, displayed in magenta
    image2 : np.array
        Image 2, displayed in green
    normalize : bool, optional
        If True, the image range is normalized by the maximum and minimum
        values in the image, by default True
    """
    if len(image1.shape) == 3:
        image1 = sk.color.rgb2gray(image1)

    if len(image2.shape) == 3:
        image2 = sk.color.rgb2gray(image2)

    # Trim images to the smaller of the two dimensions
    hf = np.min([image1.shape[0], image2.shape[0]])
    wf = np.min([image1.shape[1], image2.shape[1]])

    image1 = image1[:hf, :wf]
    image2 = image2[:hf, :wf]

    if not (image1.shape == image2.shape):
        raise ValueError(
            f"The two images are not the same shape. (Image1:{image1.shape}, Image2:{image2.shape})"
        )

    if normalize:
        # Normalize images to make sure they look good
        image1 = sk.exposure.rescale_intensity(
            image1, in_range="image", out_range=(0.0, 1.0)
        )

        image2 = sk.exposure.rescale_intensity(
            image2, in_range="image", out_range=(0.0, 1.0)
        )

    image1 = sk.util.img_as_ubyte(image1)
    image2 = sk.util.img_as_ubyte(image2)

    merged = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)

    merged[..., 0] = image1
    merged[..., 1] = image2
    merged[..., 2] = image1

    return merged


def overlay_mask(image, mask, normalize_image=True, mask_color=(0, 1, 0), alpha=0.3):
    """
    Overlay mask on image.

    Parameters
    ----------
    image : ndarray
        Grayscale or RGB image
    mask : array-like
        Binary mask or label matrix
    normalize_image : bool, optional
        If True, normalizes the image intensity to its minimum and maximum
        values to help improve the image appearance. By default, True.
    mask_color : tuple, optional
        Normalized RGB vector specifying the color for the mask, by default (0, 1, 0)
    alpha : float, optional
        Alpha value for the compositing image, by default 0.3. Increasing this number
        will make the mask more visible.

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    if not (image.shape[:2] == mask.shape):
        raise ValueError(
            f"Image and mask are not the same shape. (Image:{image.shape}, Mask:{mask.shape})"
        )

    if normalize_image:
        # Normalize images to make sure they look good
        image = sk.exposure.rescale_intensity(
            image,
            in_range=(0.3 * np.min(image), 0.7 * np.max(image)),
            out_range=(0.0, 1.0),
        )

    # Convert grayscale image into rgb
    if len(image.shape) < 3 or image.shape[2] == 1:
        image = sk.color.gray2rgb(image)

    # Merge the two images
    H, W = image.shape[:2]

    overlay = np.zeros((H, W, 3), dtype=image.dtype)

    for iC in range(3):
        overlay[..., iC] = (1 - alpha) * image[..., iC] + (
            alpha * mask * mask_color[iC]
        )

    overlay = sk.util.img_as_ubyte(overlay)

    return overlay


def get_ROI(image, downsample_factor=None):
    """
    Manually select a region of interest.

    The function will plot the image, then allow the user to use a rectangle
    selector to select a region of interest (ROI). When the ROI selection is
    ready, press ``enter`` to confirm the selection. Close the window to
    complete the selection.

    Parameters
    ----------
    image : ndarray
        Image
    downsample_factor : int, optional
        Factor to downsample image by, by default None. For large images,
        setting a downsample value will help with speed of display. The ROI
        will be scaled back to the original image size.

    Returns
    -------
    final_roi_list : list of dicts
        Keys are xmin, xmax, ymin, ymax.
    """
    all_rois = []
    current_coords = None  # Holds the unsaved box coordinates

    def onselect(eclick, erelease):
        """Update the temporary coordinates whenever a box is drawn/resized."""
        nonlocal current_coords

        xmin, xmax = (
            int(min(eclick.xdata, erelease.xdata)),
            int(max(eclick.xdata, erelease.xdata)),
        )
        ymin, ymax = (
            int(min(eclick.ydata, erelease.ydata)),
            int(max(eclick.ydata, erelease.ydata)),
        )
        current_coords = (xmin, xmax, ymin, ymax)

    def on_key(event):
        """Listen for keyboard inputs."""
        nonlocal current_coords

        if event.key == "enter":
            if current_coords is not None:
                xmin, xmax, ymin, ymax = current_coords

                # Save the coordinates as a dict
                roi = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
                all_rois.append(roi)
                print(f"ROI #{len(all_rois)}: {roi}")

                # Draw a rectangle on the image
                width = xmax - xmin
                height = ymax - ymin
                rect = Rectangle(
                    (xmin, ymin),
                    width,
                    height,
                    edgecolor="green",
                    facecolor="none",
                    linewidth=1,
                )
                ax.add_patch(rect)

                # Refresh the plot to show the new permanent patch
                fig.canvas.draw()

                # Reset temporary storage so we don't duplicate on double-enter
                current_coords = None
            else:
                print("No new ROI drawn to save!")

    # Downsize the image for easier viewing
    if downsample_factor:
        image = image[::downsample_factor, ::downsample_factor, :]

    # image = sk.exposure.rescale_intensity(image, in_range=(np.min(image), 0.5 * np.max(image)), out_range=(0.0, 1.0))

    fig, ax = plt.subplots(figsize=(12, 10))
    # ax.imshow(image, cmap="gray")
    ax.imshow(image)
    ax.set_title(
        "Drag to resize and move the selection. Press enter to create an ROI."
        "Close image when done."
    )

    fig.canvas.mpl_connect("key_press_event", on_key)

    # Enable the selector
    rs = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)

    plt.show()

    # Rescale the ROIs by the downsample factor
    if downsample_factor:
        final_roi_list = [
            {
                "xmin": roi["xmin"] * downsample_factor,
                "xmax": roi["xmax"] * downsample_factor,
                "ymin": roi["ymin"] * downsample_factor,
                "ymax": roi["ymax"] * downsample_factor,
            }
            for roi in all_rois
        ]

    else:
        final_roi_list = all_rois

    return final_roi_list
