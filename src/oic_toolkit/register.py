import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
from scipy import interpolate, ndimage, spatial
from scipy.fft import fft2, fftshift
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

from oic_toolkit._internal import validate_path

from . import display, util


def phasexcorr(target, moving, return_corrected=True):
    """
    Register images using phase cross-correlation.

    This function registers translational shifts using the phase cross-correlation
    function. Internally uses scikit-image's phase_cross_correlation function, with
    some wrapping to match image sizes and perform the correction.

    Parameters
    ----------
    target : ndarray
        The target image which acts as the reference
    moving : ndarray
        The moving image
    return_corrected : bool, optional
        If True will return the corrected moving image as a second output variable, by
        default True. If only the shift is required, it could be faster to set this to
        False.

    Returns
    -------
    results: dict
        Dictionary containing the "shift", "error" and "diffphase" of the phase
        cross-correlation.
    moving_corrected : ndarray, optional
        The corrected moving image. Only returned if `return_corrected` is True.
    """
    if len(target.shape) == 3:
        target = sk.color.rgb2gray(target)
    h0, w0 = target.shape

    if len(moving.shape) == 3:
        moving = sk.color.rgb2gray(moving)
    hm, wm = moving.shape

    # Match the moving to the target shape
    moving_final = np.zeros_like(target)

    # Determine the overlapping bounds
    slice_y = min(h0, hm)
    slice_x = min(w0, wm)

    moving_final[:slice_y, :slice_x] = moving[:slice_y, :slice_x]

    # Run the phase cross-correlation to register the images
    shift, error, diffphase = sk.registration.phase_cross_correlation(
        target, moving_final, disambiguate=True
    )

    # Combine the output metrics
    results = {"shift": shift, "error": error, "diffphase": diffphase}

    if return_corrected:
        moving_corrected = ndimage.shift(moving_final, shift=shift, cval=0.0)

        return results, moving_corrected

    else:
        return results


def shift_image(image, shift):
    """
    Translate image.

    This function translates an image (e.g., after phase cross-correlation
    registration). Internally uses the ndimage.shift function.

    Parameters
    ----------
    image : ndarray
        Image to shift
    shift : float
        Amount to shift the image by.

    Returns
    -------
    _type_
        _description_
    """
    shifted_image = ndimage.shift(image, shift=shift, cval=0.0)

    return shifted_image


# -- These functions are for template matching/displacement field registration


def calculate_displacement_field(
    target,
    moving,
    num_grid=(10, 10),
    template_size=200,
    search_window=150,
    show_plots=False,
):
    """
    Calculate a displacement field for image registration.

    This function uses a displacement field approach for image registration. In short,
    the image is divided into a grid of sub-images. Template matching is used to
    identify the translational shift of each sub-image. This generates the displacement
    field, which can then be used to warp the image.

    To limit the search time, the template matching is only carried out over a set search window.

    Parameters
    ----------
    target : ndarray
        Target image
    moving : ndarray
        Moving image
    num_grid : tuple, optional
        Size of grid to perform registration, by default (10, 10)
    template_size : int, optional
        Size of the template, by default 200
    search_window : int, optional
        Search window, by default 150
    show_plots : bool, optional
        If True, will display plots for each grid, by default False

    Returns
    -------
    src_points : ndarray
        An array containing the source coordinates.
    dst_points : ndarray
        An array containing the corresponding destination coordinates of the registered
        image.
    """
    # if search_window < template_size:
    #     raise ValueError(f"The search window must be larger than the template")

    # The basic idea is to divide up the moving image and register it to the target image. The shift defines the displacement at each grid point. In this step, we will be doing this on a low-resolution grid, which will be scaled up via interpolation later to get a smoother displacement field.
    #
    # We assume that the images are already *roughly* aligned

    # Try band-pass filtering
    target = sk.filters.difference_of_gaussians(target, 5, 20)
    moving = sk.filters.difference_of_gaussians(moving, 5, 20)

    # --- Generate the image grid ---
    # The image grid should be generated inside the actual image.
    # Calculate the margin of the image
    margin = (template_size // 2) + search_window

    H, W = target.shape
    # Define the grid against the target image because this is the size we want to
    # match to

    # Calculate the center of each grid point
    # Note: np.linspace(start, stop, num)
    y_coords = np.linspace(margin, H - margin, num_grid[0], dtype=int)
    x_coords = np.linspace(margin, W - margin, num_grid[1], dtype=int)

    # Initialize empty matrices to store data
    src_points = []
    dst_points = []

    for yy in y_coords:
        for xx in x_coords:
            # Calculate the top-left corner position of the template
            top_template = yy - template_size // 2
            left_template = xx - template_size // 2

            template = moving[
                top_template : (top_template + template_size),
                left_template : (left_template + template_size),
            ]

            # print(np.std(template))

            # Skip if there is no image information (i.e., just noise)
            if np.std(template) < 0.01:
                print("Skipping patch")
                continue

            # Define a search window for the target image
            # top_target_window = yy - search_window//2
            # left_target_window = xx - search_window//2

            # target_window = target[
            #     top_target_window:(top_target_window + search_window),
            #     left_target_window:(left_target_window + search_window)]

            wy0, wy1 = (
                max(0, top_template - search_window),
                min(H, top_template + template_size + search_window),
            )
            wx0, wx1 = (
                max(0, left_template - search_window),
                min(W, left_template + template_size + search_window),
            )

            target_window = target[wy0:wy1, wx0:wx1]

            # Perform the template matching
            try:
                template_f = sk.filters.sobel(template)
                target_window_f = sk.filters.sobel(target_window)

                corr_coeff = sk.feature.match_template(target_window_f, template_f)

            except Exception:
                continue

            # Find the highest response
            max_ij = np.unravel_index(np.argmax(corr_coeff), corr_coeff.shape)

            src_points.append([xx, yy])

            dst_points.append(
                [
                    max_ij[1] + wx0 + template_size // 2,
                    max_ij[0] + wy0 + template_size // 2,
                ]
            )
            # dst_points.append([top_template + max_ij[1],
            #                    left_template + max_ij[0]])

            if show_plots:
                # Make some plots
                fig = plt.figure(figsize=(16, 6))
                ax1 = plt.subplot(1, 3, 1)
                ax2 = plt.subplot(1, 3, 2)
                ax3 = plt.subplot(1, 3, 3)

                ax1.imshow(target_window)
                ax1.set_axis_off()
                ax1.set_title("Search window (red box indicates matched region)")

                x, y = max_ij[::-1]
                rect = plt.Rectangle(
                    (x, y),
                    template_size,
                    template_size,
                    edgecolor="r",
                    facecolor="none",
                )
                ax1.add_patch(rect)

                ax2.imshow(template)
                ax2.set_axis_off()
                ax2.set_title("Template")

                # Crop the target image around the destination to double-check
                left_crop = dst_points[-1][0]
                top_crop = dst_points[-1][1]
                target_cropped = target[
                    top_crop - (template_size // 2) : (top_crop + (template_size // 2)),
                    left_crop - (template_size // 2) : (
                        left_crop + (template_size // 2)
                    ),
                ]

                merge = display.merge_images(target_cropped, template)

                ax3.imshow(merge)
                ax3.set_title("Merged image")

                plt.show()

                plt.close(fig)

    return np.array(src_points), np.array(dst_points)


def estimate_tform(
    src, dst, target_shape, mesh_grid=(100, 100), max_distance=300, corners="median"
):

    print("Estimating final transform...")

    H, W = target_shape

    # Calculate the displacement field
    displacements = dst - src

    # Generate the source grid (over the target image)
    grid_y, grid_x = np.mgrid[0 : H : mesh_grid[0], 0 : W : mesh_grid[1]]
    full_src_grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Interpolate the displacement field over the entire mesh. This helps to
    # smooth variations in the displacement field and avoid "tearing" the image. The function uses the median value as a fallback for regions where the interpolation fails (typically along the image border).
    interp_dx = interpolate.griddata(
        src,
        displacements[:, 0],
        full_src_grid,
        method="linear",
        fill_value=np.median(displacements[:, 0]),
    )
    interp_dy = interpolate.griddata(
        src,
        displacements[:, 1],
        full_src_grid,
        method="linear",
        fill_value=np.median(displacements[:, 1]),
    )

    global_median = np.median(displacements, axis=0)

    if max_distance is not None:
        print("Applying distance-based stability mask...")
        tree = spatial.cKDTree(src)
        distances, _ = tree.query(full_src_grid)

        alpha = np.clip(1 - (distances / max_distance), 0, 1)
        final_dx = (interp_dx * alpha) + (global_median[0] * (1 - alpha))
        final_dy = (interp_dy * alpha) + (global_median[1] * (1 - alpha))
    else:
        final_dx = interp_dx
        final_dy = interp_dy

    # Reconstruct the full destination grid
    full_dst_grid = full_src_grid + np.vstack([final_dx, final_dy]).T

    match corners:
        case "static":
            # Add static corner anchors to keep the frame square
            anchors = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]])
            full_src_grid = np.vstack([full_src_grid, anchors])
            full_dst_grid = np.vstack([full_dst_grid, anchors])

        case "median":
            anchors_src = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]])
            anchors_dst = anchors_src + global_median

            full_src_grid = np.vstack([full_src_grid, anchors_src])
            full_dst_grid = np.vstack([full_dst_grid, anchors_dst])

    # Apply Piecewise Affine Transform to calculate the final warp matrix
    # tform = sk.transform.PiecewiseAffineTransform.from_estimate(full_dst_grid, full_src_grid)
    print("Done.")

    return _, full_src_grid, full_dst_grid


def fast_warp(
    image_to_warp, target_shape, full_src_grid, full_dst_grid, mesh_grid=(100, 100)
):
    """
    Warp an image using fast coordinate mapping.

    Instantly warps an image using dense coordinate mapping instead of
    slow pixel-by-pixel Delaunay triangulation.

    Parameters
    ----------
    image_to_warp : ndarray
        The source image you want to correct/warp (e.g., I2).
    target_shape : tuple
        The (Height, Width) of the reference image grid (e.g., I1.shape[:2]).
    full_src_grid : ndarray
        The source grid returned by estimate_tform.
    full_dst_grid : ndarray
        The destination grid returned by estimate_tform.
    mesh_grid : tuple, optional
        The step sizes used to build the original grid. Must match estimate_tform.

    Returns
    -------
    corrected : ndarray
        The warped image with identical spatial dimensions to target_shape.
    """
    H, W = target_shape[:2]

    # 1. Exclude the 4 corner anchors added at the end of estimate_tform
    num_grid_points = full_src_grid.shape[0] - 4
    dst_pts = full_dst_grid[:num_grid_points]

    # 2. Reconstruct the original 2D sparse grid dimensions
    grid_y, grid_x = np.mgrid[0 : H : mesh_grid[0], 0 : W : mesh_grid[1]]
    grid_shape = grid_y.shape

    # 3. Reshape flat grid coordinates back into 2D maps (index 0 is X, index 1 is Y)
    map_x_sparse = dst_pts[:, 0].reshape(grid_shape)
    map_y_sparse = dst_pts[:, 1].reshape(grid_shape)

    # 4. Upsample the sparse grid coordinate maps to the full target image resolution
    map_x_full = sk.transform.resize(map_x_sparse, (H, W), order=1, mode="edge")
    map_y_full = sk.transform.resize(map_y_sparse, (H, W), order=1, mode="edge")

    # 5. Stack coordinates in [Y, X] order required by scipy's map_coordinates
    coords = np.array([map_y_full, map_x_full])

    # 6. Map the pixels instantly (handling color vs grayscale)
    if image_to_warp.ndim == 3:
        corrected = np.empty((H, W, image_to_warp.shape[2]), dtype=image_to_warp.dtype)
        for c in range(image_to_warp.shape[2]):
            corrected[..., c] = ndimage.map_coordinates(
                image_to_warp[..., c], coords, order=1, mode="constant", cval=0
            )
    else:
        corrected = ndimage.map_coordinates(
            image_to_warp, coords, order=1, mode="constant", cval=0
        )

    return corrected


def generate_quiver_plot(target, src, dst):
    """
    Generate a quiver plot to visualize displacement field.

    This plot can be helpful when trying to validate a displacement field registration.
    If the registration is successful, the quiver plot should appear smoothly varying.
    If not, the arrows will point in all directions.

    Parameters
    ----------
    target : ndarray
        Target image.
    src : ndarray
        An array containing the source coordinates.
    dst : ndarray
        An array containing the destination coordinates.
    """
    plt.figure(figsize=(15, 15))
    plt.imshow(target, cmap="gray", alpha=0.7)

    # Calculate displacement vectors
    dx = dst[:, 0] - src[:, 0]
    dy = dst[:, 1] - src[:, 1]

    # Plot original raw vectors in Red
    plt.quiver(
        src[:, 0],
        src[:, 1],
        dx,
        dy,
        color="red",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.002,
        label="Raw Matches",
    )
    plt.show()


def rotation_scale_phasecorr(target, moving, radius=None):
    """
    Corrects rotation and scale using phase correlation.

    This function uses log-polar transformations to identify rotation and scale
    differences using the fast phase correlation algorithm. Note that this function
    currently assumes that the center of the transformation is the center of the image.

    Parameters
    ----------
    target : ndarray
        Target image
    moving : ndarray
        Moving image
    radius : float, optional
        Radius of the circle that bounds the area to be transformed, by default None

    Returns
    -------
    rotation_in_degrees : float
        Registered rotation in degrees
    shift_scale : float
        Registered image scaling
    corrected_image: ndarray
        The corrected image
    """
    # Estimate radius to be the smallest dimension
    h, w = target.shape[:2]

    if not radius:
        radius = min(h, w) // 2

    target_polar = sk.transform.warp_polar(target, radius=radius, scaling="log")
    moving_polar = sk.transform.warp_polar(moving, radius=radius, scaling="log")

    shifts, error, phasediff = sk.registration.phase_cross_correlation(
        target_polar, moving_polar, upsample_factor=20, normalization=None
    )
    shiftr, shiftc = shifts[:2]

    # Calculate scale factor from translation
    klog = radius / np.log(radius)
    shift_scale = 1 / (np.exp(shiftc / klog))

    # Note: shiftr is the number of rows (in polar coordinates, this is the same as angle)
    rotation_in_degrees = (shiftr / target_polar.shape[0]) * 360.0
    rotation_in_radians = np.deg2rad(rotation_in_degrees)

    # Calculate the corrected image
    # Note: SimilarityTransform uses the top left pixel as the center of rotation, while rotate() uses the center of the image. So you have to translate the image before rotating, then transfer back
    center = np.array([w / 2, h / 2])

    t1 = sk.transform.SimilarityTransform(translation=-center)
    t2 = sk.transform.SimilarityTransform(
        rotation=-rotation_in_radians, scale=shift_scale
    )
    t3 = sk.transform.SimilarityTransform(translation=center)

    # Combine the tforms
    tform = t1 + t2 + t3
    corrected_image = sk.transform.warp(moving, tform)

    return rotation_in_degrees, shift_scale, corrected_image


def plot_diff_gauss(image):
    sigmas = [(1, 5), (3, 15), (5, 20), (5, 50)]

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, (s_low, s_high) in zip(axes, sigmas):
        bpf = sk.filters.difference_of_gaussians(image, s_low, s_high)
        ax.imshow(bpf, cmap="gray")
        ax.set_title(f"Low: {s_low}, High: {s_high}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def log_polar_phasecorr(target, moving, sigma_low=5, sigma_high=20, window_size=None):

    # See:https://scikit-image.org/docs/stable/auto_examples/registration/plot_register_rotation.html
    target, moving = pad_images(target, moving)

    # Band pass the images
    target_bpf = sk.filters.difference_of_gaussians(target, sigma_low, sigma_high)
    moving_bpf = sk.filters.difference_of_gaussians(moving, sigma_low, sigma_high)

    # Window
    if window_size is None:
        window_size = target.shape

    window = sk.filters.window("hann", window_size)
    window_padded = window

    _, window_padded = pad_images(target, window)

    target_bpf_w = target_bpf * window_padded
    moving_bpf_w = moving_bpf * window_padded

    # plt.imshow(target_bpf_w)
    # plt.show()
    # exit()

    # FFT shift
    target_fs = np.abs(fftshift(fft2(target_bpf_w)))
    moving_fs = np.abs(fftshift(fft2(moving_bpf_w)))

    radius = target_bpf_w.shape[0] // 8  # Take only the lower frequencies

    fs_shape = target.shape
    h, w = fs_shape

    warped_target_fs = sk.transform.warp_polar(
        target_fs, radius=radius, output_shape=fs_shape, scaling="log", order=0
    )
    warped_moving_fs = sk.transform.warp_polar(
        moving_fs, radius=radius, output_shape=fs_shape, scaling="log", order=0
    )

    warped_target_fs = warped_target_fs[: h // 2, :]  # only use half of FFT
    warped_moving_fs = warped_moving_fs[: h // 2, :]

    # Try masking the images
    # mask_target = target > 0
    # mask_moving = moving > 0

    shifts, error, phasediff = sk.registration.phase_cross_correlation(
        warped_target_fs,
        warped_moving_fs,
        upsample_factor=1000,
        normalization=None,
        # reference_mask=mask_target, moving_mask=mask_moving
    )

    # Use translation parameters to calculate rotation and scaling parameters
    shiftr, shiftc = shifts[:2]

    rotation_in_degrees = (shiftr / h) * 360.0

    klog = w / np.log(radius)
    shift_scale = np.exp(shiftc / klog)

    rotation_in_radians = np.deg2rad(rotation_in_degrees)

    # Calculate the corrected image
    center = np.array([w / 2, h / 2])

    t1 = sk.transform.SimilarityTransform(translation=-center)
    t2 = sk.transform.SimilarityTransform(
        rotation=-rotation_in_radians, scale=shift_scale
    )
    t3 = sk.transform.SimilarityTransform(translation=center)

    # Combine the tforms
    tform = t1 + t2 + t3
    corrected_image = sk.transform.warp(moving, tform)

    # # Register the image to get the translation
    # shift_translation, error_translation, phasediff_translation = sk.registration.phase_cross_correlation(
    #     target, corrected_image, upsample_factor=100)

    # shift_y, shift_x = shift_translation

    # tform_translate = sk.transform.SimilarityTransform(translation=(shift_x, shift_y))
    # corrected_image_final = sk.transform.warp(corrected_image, tform_translate.inverse)

    # shift_out = (shift_x, shift_y)

    # return rotation_in_degrees, shift_scale, shift_out, corrected_image_final

    return rotation_in_degrees, shift_scale, corrected_image


def pad_images(image1, image2):
    """
    Pad images to be the same size.

    This function pads the input images appropriately to be the same output size. The
    output size will be the largest of either dimension of the two images.

    Parameters
    ----------
    image1 : ndarray
        First image
    image2 : ndarray
        Second image

    Returns
    -------
    image1_out : ndarray
        Output first image
    image2_out : ndarray
        Output second image
    """
    # Find the largest of the two images and pad the images centrally
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # print(image1.shape)
    # print(image2.shape)

    h_out = np.max([h1, h2])
    w_out = np.max([w1, w2])

    image1_out = np.ones((h_out, w_out), dtype=image1.dtype) * np.median(image1)

    h1_left = (w_out - w1) // 2
    h1_top = (h_out - h1) // 2

    image1_out[h1_top : (h1_top + h1), h1_left : (h1_left + w1)] = image1

    image2_out = np.ones((h_out, w_out), dtype=image2.dtype) * np.median(image2)
    h2_left = (w_out - w2) // 2
    h2_top = (h_out - h2) // 2

    image2_out[h2_top : (h2_top + h2), h2_left : (h2_left + w2)] = image2

    # print(image1_out.shape)
    # print(image2_out.shape)

    return image1_out, image2_out


def match_size(target, moving):
    """
    Adjust the moving image to the target image.

    If the target image was larger, the moving image will be padded. Otherwise, the
    moving image will be cropped. The operations are carried out from the center of the
    original image.

    Parameters
    ----------
    target : ndarray
        Target image to match to
    moving : ndarray
        Moving image. Note that the moving image

    Returns
    -------
    moving_out : ndarray
        Moving image with the same size as the target image.
    """
    # TODO: Handle if the images are different ranges (e.g., one is uint8 ad the other
    # is float)

    ht, wt = target.shape[:2]
    hm, wm = moving.shape[:2]

    # moving_out = np.zeros_like(target)
    moving_out = np.zeros((ht, wt) + moving.shape[2:], dtype=moving.dtype)

    # print(f"Output shape: {moving_out.shape}")
    # print(f"Output dtype: {moving_out.dtype}")

    if hm <= ht:
        # Move image down
        hdiff = (ht - hm) // 2

        hstart_out = hdiff
        hend_out = hdiff + hm

        hstart_in = 0
        hend_in = hm

    elif hm > ht:
        hdiff = (hm - ht) // 2

        hstart_out = 0
        hend_out = ht

        hstart_in = hdiff
        hend_in = hdiff + ht

    if wm <= wt:
        # Move image right
        wdiff = (wt - wm) // 2

        wstart_out = wdiff
        wend_out = wdiff + wm

        wstart_in = 0
        wend_in = wm

    elif wm > wt:
        wdiff = (wm - wt) // 2

        wstart_out = 0
        wend_out = wm

        wstart_in = wdiff
        wend_in = wdiff + wt

    moving_out[hstart_out:hend_out, wstart_out:wend_out, ...] = moving[
        hstart_in:hend_in, wstart_in:wend_in, ...
    ]

    return moving_out


def optical_flow_tvl1(target, moving):

    # Get the SMALLEST image size to prevent shearing
    ht, wt = target.shape
    hm, wm = moving.shape

    h = min(ht, hm)
    w = min(wt, wm)

    # TODO: Try masking the target and moving images - this is not available for the optical flow registration

    target = util.resize_from_center(target, (h, w))
    moving = util.resize_from_center(moving, (h, w))

    v, u = sk.registration.optical_flow_tvl1(
        target, moving, attachment=5, tightness=0.2, prefilter=True
    )

    return u, v


def correct_optical_flow(image, u, v):

    nr, nc = image.shape[:2]

    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")

    if len(image.shape) > 2:
        num_channels = image.shape[2]

        corrected = np.zeros_like(image)
        for iC in range(num_channels):
            corrected[:, :, iC] = sk.transform.warp(
                image[:, :, iC], np.array([row_coords + v, col_coords + u]), mode="edge"
            )

    else:
        corrected = sk.transform.warp(
            image, np.array([row_coords + v, col_coords + u]), mode="edge"
        )

    return corrected


# --- Stitching ---#
def stitch_xy(image_path, numX, numY, overlap=15, file_pattern="img_{ii}.tif"):

    # Validate image_path
    image_path = validate_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Directory {image_path} does not exist.")

    # Check that the files exist
    file_pattern_r = file_pattern.replace("{ii}", "[0-9][0-9]")

    file_list = sorted(list(image_path.glob(file_pattern_r)))

    if len(file_list) == 0:
        raise FileNotFoundError(
            f"No files matching the pattern '{file_pattern}' was found."
        )
    elif len(file_list) != (numX * numY):
        raise FileNotFoundError(
            f"Expected {numX * numY} files, but found only {len(file_list)}."
        )

    # Load one file to get dimensions for the stitched image
    image = sk.io.imread(file_list[0])

    tile_h, tile_w = image.shape[:2]
    dtype = image.dtype

    # print(dtype)
    # exit()

    # To avoid needing to load ALL the images into memory at once, the plan is to load
    # two rows. Once the top row is registered to the bottom, it can be popped off and
    # the top row is now the bottom row and the code can continue.

    tile_cache = {}

    H_displacements = {}
    V_displacements = {}

    for row in range(numY):
        for col in range(numX):
            curr_tile_idx = (row * numX) + col

            print(f"({row}, {col}): {curr_tile_idx}")

            if curr_tile_idx not in tile_cache:
                tile_cache[curr_tile_idx] = sk.io.imread(file_list[curr_tile_idx])

            if col < (numX - 1):
                # Load the right tile
                right_tile_idx = curr_tile_idx + 1

                if right_tile_idx not in tile_cache:
                    tile_cache[right_tile_idx] = sk.io.imread(file_list[right_tile_idx])

                H_displacements[(col, row)] = _match_edges(
                    tile_cache[curr_tile_idx],
                    tile_cache[right_tile_idx],
                    direction="h",
                    overlap_percent=overlap,
                )

            if row < (numY - 1):
                bottom_tile_idx = curr_tile_idx + numX
                if bottom_tile_idx not in tile_cache:
                    tile_cache[bottom_tile_idx] = sk.io.imread(
                        file_list[bottom_tile_idx]
                    )

                V_displacements[(col, row)] = _match_edges(
                    tile_cache[curr_tile_idx],
                    tile_cache[bottom_tile_idx],
                    direction="v",
                    overlap_percent=overlap,
                )

        if row > 0:
            for col in range(numX):
                idx_to_pop = (row - 1) * numX + col
                tile_cache.pop(idx_to_pop, None)

    # Clear out remaining tiles from registration phase
    tile_cache.clear()

    # --- PASS 2: Global Optimization Solver ---
    abs_x, abs_y = _solve_global_positions(numX, numY, H_displacements, V_displacements)

    # --- PASS 3: Generate Canvas and Stream/Blend Images One-by-One ---
    canvas_w = int(np.max(abs_x) + tile_w)
    canvas_h = int(np.max(abs_y) + tile_h)

    print(f"Canvas size: {canvas_w}x{canvas_h}")

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    weight_canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    # Generate linear alpha feathering masks based on your overlap setting
    overlap_frac = (overlap / 100.0) if overlap > 1 else overlap
    ov_x, ov_y = int(tile_w * overlap_frac), int(tile_h * overlap_frac)

    ramp_y = np.ones(tile_h, dtype=np.float32)
    ramp_x = np.ones(tile_w, dtype=np.float32)
    if ov_y > 0:
        ramp_y[:ov_y] = np.linspace(0, 1, ov_y)
        ramp_y[-ov_y:] = np.linspace(1, 0, ov_y)
    if ov_x > 0:
        ramp_x[:ov_x] = np.linspace(0, 1, ov_x)
        ramp_x[-ov_x:] = np.linspace(1, 0, ov_x)
    tile_mask = np.outer(ramp_y, ramp_x)

    # Blend target frames directly onto normalized matrix map
    for idx, file_path in enumerate(file_list):
        tile = sk.io.imread(file_path).astype(np.float32)
        x_start, y_start = int(abs_x[idx]), int(abs_y[idx])

        canvas[y_start : y_start + tile_h, x_start : x_start + tile_w] += (
            tile * tile_mask
        )
        weight_canvas[y_start : y_start + tile_h, x_start : x_start + tile_w] += (
            tile_mask
        )

    weight_canvas[weight_canvas == 0] = 1.0
    final_canvas = canvas / weight_canvas

    max_val = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    return np.clip(final_canvas, 0, max_val).astype(dtype)


def _match_edges(tile_a, tile_b, overlap_percent, direction="h"):
    h, w = tile_a.shape[:2]

    if overlap_percent > 1:
        overlap_percent = overlap_percent / 100

    overlap_x = int(overlap_percent * w)
    overlap_y = int(overlap_percent * h)

    match direction.lower():
        case "h":
            # Assume that tile_b is to the RIGHT of tile_a
            crop_a = tile_a[:, (w - overlap_x) :]
            crop_b = tile_b[:, :overlap_x]

            # Sanity check
            if crop_a.shape != crop_b.shape:
                print(crop_a.shape)
                print(crop_b.shape)
                raise ValueError("The matrices are different shapes")

            # Run the cross-correlation
            shift, _, _ = sk.registration.phase_cross_correlation(
                crop_a, crop_b, upsample_factor=8
            )

            dx = (w - overlap_x) + shift[1]
            dy = shift[0]
            return dx, dy

        case "v":
            # Assume that tile_b is BELOW tile_a
            crop_a = tile_a[(h - overlap_y) :, :]
            crop_b = tile_b[:overlap_y, :]

            # Sanity check
            if crop_a.shape != crop_b.shape:
                print(crop_a.shape)
                print(crop_b.shape)
                raise ValueError("The matrices are different shapes")

            # Run the cross-correlation
            shift, _, _ = sk.registration.phase_cross_correlation(
                crop_a, crop_b, upsample_factor=8
            )

            dx = shift[1]
            dy = (h - overlap_y) + shift[0]
            return dx, dy


def _solve_global_positions(numX, numY, H_displacements, V_displacements):
    """
    Solves a least-squares matrix system mapping relative tile shifts
    to absolute (X, Y) canvas coordinates.
    """
    n_tiles = numX * numY
    n_constraints = (numX - 1) * numY + numX * (numY - 1)

    # Initialize sparse linear equations matrices: A * coordinates = b
    A = lil_matrix((n_constraints + 1, n_tiles), dtype=np.float32)
    b_x = np.zeros(n_constraints + 1, dtype=np.float32)
    b_y = np.zeros(n_constraints + 1, dtype=np.float32)

    eq_idx = 0
    # Map Horizontal constraints: tile(x+1) - tile(x) = dx
    for row in range(numY):
        for col in range(numX - 1):
            idx1 = row * numX + col
            idx2 = idx1 + 1
            A[eq_idx, idx1] = -1
            A[eq_idx, idx2] = 1
            b_x[eq_idx], b_y[eq_idx] = H_displacements[(col, row)]
            eq_idx += 1

    # Map Vertical constraints: tile(y+1) - tile(y) = dy
    for row in range(numY - 1):
        for col in range(numX):
            idx1 = row * numX + col
            idx2 = idx1 + numX
            A[eq_idx, idx1] = -1
            A[eq_idx, idx2] = 1
            b_x[eq_idx], b_y[eq_idx] = V_displacements[(col, row)]
            eq_idx += 1

    # Pin anchor frame (0,0) down to absolute coordinate (0,0)
    A[eq_idx, 0] = 1
    b_x[eq_idx], b_y[eq_idx] = 0.0, 0.0

    # Solve system using Sparse Least Squares
    sol_x = lsqr(A.tocsr(), b_x)[0]
    sol_y = lsqr(A.tocsr(), b_y)[0]

    # Normalize coordinate origin to zero
    sol_x -= np.min(sol_x)
    sol_y -= np.min(sol_y)

    return sol_x, sol_y
