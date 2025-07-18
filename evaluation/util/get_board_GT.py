import numpy as np
import cv2

def look_at_rotation(camera_position, target=np.array([0, 0, 0]), up=np.array([0, 0, -1])):
    """
    Computes a camera-to-world rotation matrix so that the camera at 'camera_position' looks at 'target',
    using a consistent 'up' vector.
    
    Returns:
        R_c2w: 3x3 rotation matrix (camera-to-world).
    """
    # Forward direction (camera Z axis, pointing towards target)
    z_axis = (target - camera_position)
    z_axis /= np.linalg.norm(z_axis)

    # Project the up vector onto the plane orthogonal to z
    up_proj = up - np.dot(up, z_axis) * z_axis
    up_proj /= np.linalg.norm(up_proj)

    # Right (camera X axis)
    x_axis = np.cross(up_proj, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    # True up (camera Y axis)
    y_axis = np.cross(z_axis, x_axis)

    # Build rotation matrix (camera-to-world)
    R_c2w = np.stack([x_axis, y_axis, z_axis], axis=1)

    return R_c2w

def get_projected_rectangle_corners(x_l, y_l, z_l, R, T, K):
    """
    Projects the corners of a rectangle at the origin into image coordinates.
    """
    corners_world = np.array([
        [-x_l / 2, -y_l / 2,  z_l / 2],
        [ x_l / 2, -y_l / 2,  z_l / 2],
        [ x_l / 2,  y_l / 2,  z_l / 2],
        [-x_l / 2,  y_l / 2,  z_l / 2],
    ])  # Shape (4, 3)

    # Convert world -> camera coordinates
    corners_cam = (R @ (corners_world - T).T).T  # (4, 3)

    # Project to image
    corners_proj = (K @ corners_cam.T).T
    corners_img = corners_proj[:, :2] / corners_cam[:, 2][:, np.newaxis]

    return corners_img.tolist()

def draw_rectangle_on_image(image, projected_points, color=(0, 255, 0), thickness=2):
    img = image.copy()
    pts = np.round(projected_points).astype(int)
    for i in range(4):
        pt1 = tuple(pts[i])
        pt2 = tuple(pts[(i + 1) % 4])
        cv2.line(img, pt1, pt2, color, thickness)
    return img

def warp_to_square(image, pts, output_size=256):
    """
    Warps the region inside the given four corner points into a square.

    Args:
        image: Input image.
        pts: Four 2D points (x, y) defining the quadrilateral (TL, TR, BR, BL).
        output_size: Size (in pixels) of the output square (default 256x256).

    Returns:
        Warped square image.
    """
    # Convert to float32 numpy array
    pts = np.array(pts, dtype="float32")

    # Define the destination square
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)

    # Warp the image
    warped = cv2.warpPerspective(image, M, (output_size, output_size))

    return warped, M, dst  # return matrix and dst in case needed later

def color_grid_on_square(square_img, grid_size=8):
    h, w = square_img.shape[:2]
    cell_w = w // grid_size
    cell_h = h // grid_size

    # Draw checkerboard over copy
    colored = square_img.copy()
    for y in range(grid_size):
        for x in range(grid_size):
            color = (x, y, 0)
            cv2.rectangle(
                colored,
                (x * cell_w, y * cell_h),
                ((x + 1) * cell_w, (y + 1) * cell_h),
                color,
                thickness=-1
            )
    return colored

def warp_square_back(original_image, modified_square, src_quad, square_size=256):
    src_quad = np.array(src_quad, dtype="float32")
    dst_square = np.array([
        [0, 0],
        [square_size - 1, 0],
        [square_size - 1, square_size - 1],
        [0, square_size - 1]
    ], dtype="float32")

    Minv = cv2.getPerspectiveTransform(dst_square, src_quad)

    # Warp back to original image
    warped_back = cv2.warpPerspective(modified_square, Minv, (original_image.shape[1], original_image.shape[0]))

    # Create overlay mask
    mask = np.any(warped_back != [0, 0, 0], axis=-1)
    output = np.zeros_like(original_image)
    output[mask] = warped_back[mask]

    return output


if __name__ == "__main__":
    image = cv2.imread("000016.png")

    # Camera intrinsics
    K = np.array([
        [888.88909234, 0, 319.5],
        [0, 888.88909234, 239.5],
        [0, 0, 1]
    ])

    # Camera position in world coordinates
    camera_position = np.array([
        2.8506911582739054,
        2.402565526143847,
        3.3965269803371374            
    ])

    # Compute rotation matrix so camera looks at the origin
    R_c2w = look_at_rotation(camera_position)
    R = R_c2w.T  # Transpose to get world-to-camera rotation
    T = camera_position  # Translation is the camera position
    print(R)
    # Rectangle dimensions (centered at origin)
    r = 2
    x_l, y_l, z_l = r, r, 0

    # Project corners
    projected_pts = get_projected_rectangle_corners(x_l, y_l, z_l, R, T, K)
    print("Projected points:", projected_pts)

    output_img = image.copy()
    for p in projected_pts:
        cv2.circle(output_img, (int(round(p[0])), int(round(p[1]))), 8, (255, 0, 0), -1)

    # Draw origin (0, 0, 0) projection
    origin_world = np.array([[0, 0, 0]])
    origin_cam = (R @ (origin_world - T).T).T
    origin_proj = (K @ origin_cam.T).T
    origin_img = origin_proj[:, :2] / origin_cam[:, 2][:, np.newaxis]
    origin_pt = origin_img[0]

    cv2.circle(output_img, (int(round(origin_pt[0])), int(round(origin_pt[1]))), 8, (255, 0, 0), -1)
    # Step 1: Warp to square
    warped_square, M, dst = warp_to_square(output_img, projected_pts, output_size=256)

    # Step 2: Color the square
    colored_square = color_grid_on_square(warped_square, grid_size=8)

    # Step 3: Warp back to original
    final_result = warp_square_back(output_img, colored_square, projected_pts, square_size=256)

    # Show result
    # Show result
    cv2.imshow("Projected Corners", output_img)
    cv2.imshow("Perspective Transform", warped_square)
    cv2.imshow("Final Overlay", final_result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
