import numpy as np
import skimage as sk
from matplotlib import pyplot as plt
from scipy import interpolate, ndimage, spatial


def register_phasexcorr(target, moving, return_corrected=True):
    """
    Attempts to register translational shifts using the phase cross-correlation function. Internally uses scikit-image's phase_cross_correlation function, with some wrapping to match image sizes and perform the correction.

    Parameters
    ----------
    target : ndarray
        The target image which acts as the reference
    moving : ndarray
        The moving image
    return_corrected : bool, optional
        If True will return the corrected moving image as a second output variable, by default True. If only the shift is required, it could be faster to set this to False.

    Returns
    -------
    results: dict
        Dictionary containing the "shift", "error" and "diffphase" of the phase cross-correlation.
    moving_corrected : ndarray, optional
        The corrected moving image. Only returned if `return_corrected` is True.
    """

    h0, w0 = target.shape
    hm, wm = moving.shape

    # Match the moving to the target shape
    moving_final = np.zeros_like(target)

    # Determine the overlapping bounds
    slice_y = min(h0, hm)
    slice_x = min(w0, wm)

    moving_final[:slice_y, :slice_x] = moving[:slice_y, :slice_x]

    # Run the phase cross-correlation to register the images
    shift, error, diffphase = sk.registration.phase_cross_correlation(
        target, 
        moving_final, 
        disambiguate=True
    )

    # Combine the output metrics
    results = {
        "shift": shift,
        "error": error,
        "diffphase": diffphase
    }

    if return_corrected:
        
        moving_corrected = ndimage.shift(
            moving_final, 
            shift=shift, 
            cval=0.0
        )

        return results, moving_corrected
    
    else:

        return results

def shift_image(image, shift):

    shifted_image = ndimage.shift(
                       image, 
                       shift=shift, 
                       cval=0.0)

    return shifted_image


#-- These functions are for template matching/displacement field registration

def calculate_displacement_field(target, moving, grid_size=(10, 10), template_size=200, search_window=100):

    H, W = target.shape
    
    # Generate the image grid
    margin = template_size // 2 + search_window

    #np.linspace(start, stop, num)
    y_coords = np.linspace(margin, H - margin, grid_size[0], dtype=int)
    x_coords = np.linspace(margin, W - margin, grid_size[1], dtype=int)

    src_points = []
    dst_points = []

    for yy in y_coords:
        for xx in x_coords:

            # Calculate the top-left corner position
            y0, x0 = yy - template_size//2, xx - template_size//2

            template = moving[y0:y0 + template_size,
                              x0:x0 + template_size]
            
            # Skip patch if there is no image information
            if np.std(template) < 0.005:
                continue

            # Define the search window coordinates to reduce computation time 
            # and avoid off-target matches. The window is clipped to the image
            # size.
            wy0, wy1 = max(0, y0 - search_window), min(H, y0 + template_size + search_window)
            wx0, wx1 = max(0, x0 - search_window), min(W, x0 + template_size + search_window)

            # Perform the template matching
            try:
                corr_coeff = sk.feature.match_template(target[wy0:wy1, wx0:wx1], template)

            except Exception:
                continue

            # Find the highest response
            max_ij = np.unravel_index(np.argmax(corr_coeff), corr_coeff.shape)

            src_points.append([xx, yy])  #Note: Points with no image data are skipped
            dst_points.append([max_ij[1] + wx0 + template_size//2, max_ij[0] + wy0 + template_size//2])

            # # Make some plots
            # fig = plt.figure(figsize=(8, 3))
            # ax1 = plt.subplot(1, 2, 1)
            # ax2 = plt.subplot(1, 2, 2)

            # ax1.imshow(target[wy0:wy1, wx0:wx1])
            # ax1.set_axis_off()

            # ax2.imshow(template)
            # ax2.set_axis_off()

            # x, y = max_ij[::-1]
            # rect = plt.Rectangle((x, y), template_size, template_size, edgecolor='r', facecolor='none')
            # ax1.add_patch(rect)
            # print(x, y)

            # plt.show()

        
    return np.array(src_points), np.array(dst_points)

def estimate_tform(src, dst, target_shape, mesh_grid=(100, 100), max_distance=300, corners="median"):

    print("Estimating final transform...")

    H, W = target_shape
    
    # Calculate the displacement field
    displacements = dst - src 
    
    # Generate the source grid (over the target image)
    grid_y, grid_x = np.mgrid[0:H:mesh_grid[0], 0:W:mesh_grid[1]]
    full_src_grid = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Interpolate the displacement field over the entire mesh. This helps to 
    # smooth variations in the displacement field and avoid "tearing" the image. The function uses the median value as a fallback for regions where the interpolation fails (typically along the image border).
    interp_dx = interpolate.griddata(src, displacements[:, 0], full_src_grid, 
                         method='linear', fill_value=np.median(displacements[:, 0]))
    interp_dy = interpolate.griddata(src, displacements[:, 1], full_src_grid, 
                         method='linear', fill_value=np.median(displacements[:, 1]))
    
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
            anchors = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]])
            full_src_grid = np.vstack([full_src_grid, anchors])
            full_dst_grid = np.vstack([full_dst_grid, anchors])

        case "median":
            anchors_src = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]])
            anchors_dst = anchors_src + global_median
            
            full_src_grid = np.vstack([full_src_grid, anchors_src])
            full_dst_grid = np.vstack([full_dst_grid, anchors_dst])

    # Apply Piecewise Affine Transform to calculate the final warp matrix
    # tform = sk.transform.PiecewiseAffineTransform.from_estimate(full_dst_grid, full_src_grid)
    print("Done.")
    
    return _, full_src_grid, full_dst_grid

def fast_warp(image_to_warp, target_shape, full_src_grid, full_dst_grid, mesh_grid=(20, 20)):
    """
    Instantly warps an image using dense coordinate mapping instead of 
    slow pixel-by-pixel Delaunay triangulation.
    
    Parameters:
    -----------
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
        
    Returns:
    --------
    corrected : ndarray
        The warped image with identical spatial dimensions to target_shape.
    """
    H, W = target_shape[:2]
    
    # 1. Exclude the 4 corner anchors added at the end of estimate_tform
    num_grid_points = full_src_grid.shape[0] - 4
    dst_pts = full_dst_grid[:num_grid_points]
    
    # 2. Reconstruct the original 2D sparse grid dimensions
    grid_y, grid_x = np.mgrid[0:H:mesh_grid[0], 0:W:mesh_grid[1]]
    grid_shape = grid_y.shape
    
    # 3. Reshape flat grid coordinates back into 2D maps (index 0 is X, index 1 is Y)
    map_x_sparse = dst_pts[:, 0].reshape(grid_shape)
    map_y_sparse = dst_pts[:, 1].reshape(grid_shape)
    
    # 4. Upsample the sparse grid coordinate maps to the full target image resolution
    map_x_full = sk.transform.resize(map_x_sparse, (H, W), order=1, mode='edge')
    map_y_full = sk.transform.resize(map_y_sparse, (H, W), order=1, mode='edge')
    
    # 5. Stack coordinates in [Y, X] order required by scipy's map_coordinates
    coords = np.array([map_y_full, map_x_full])
    
    # 6. Map the pixels instantly (handling color vs grayscale)
    if image_to_warp.ndim == 3:
        corrected = np.empty((H, W, image_to_warp.shape[2]), dtype=image_to_warp.dtype)
        for c in range(image_to_warp.shape[2]):
            corrected[..., c] = ndimage.map_coordinates(
                image_to_warp[..., c], coords, order=1, mode='constant', cval=0
            )
    else:
        corrected = ndimage.map_coordinates(
            image_to_warp, coords, order=1, mode='constant', cval=0
        )
        
    return corrected

def generate_quiver_plot(target, src, dst):
    plt.figure(figsize=(15, 15))
    plt.imshow(target, cmap='gray', alpha=0.7)
    
    # Calculate displacement vectors
    dx = dst[:, 0] - src[:, 0]
    dy = dst[:, 1] - src[:, 1]
    
    # Plot original raw vectors in Red
    plt.quiver(src[:, 0], src[:, 1], dx, dy, 
               color='red', angles='xy', scale_units='xy', scale=1, 
               width=0.002, label='Raw Matches')
    plt.show()
