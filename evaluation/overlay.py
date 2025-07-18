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

    # Show result
    cv2.imshow("Projected Rectangle", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
