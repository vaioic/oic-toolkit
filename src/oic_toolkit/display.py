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
