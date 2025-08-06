import json
import numpy as np
from collections import defaultdict
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
    print(up_proj)
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

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def getBbox(imageId): # will need to add params and adapt to fit other files
    data = load_json("/Users/rolandspoofy/developer/chess-datagen/gen-data/data_map/annotations/coco_annotations.json")

    # Build a lookup from category ID → category name
    category_map = {cat["id"]: cat["name"] for cat in data["categories"]}
    print(category_map)
    # Find the board’s category ID
    board_cat_id = next(
        (cid for cid, name in category_map.items() if name.lower() == "board"),
        None
    )
    if board_cat_id is None:
        raise ValueError("No category named 'Board' found.")

    piece_bboxes = defaultdict(list)

    for ann in data["annotations"]:
        if ann["image_id"] != imageId: # need to swap 0 with other id later
            continue

        if ann["category_id"] == board_cat_id:
            board_bbox = ann["bbox"]
        else:
            piece_name = category_map[ann["category_id"]]
            piece_bboxes[piece_name].append(ann["bbox"])
    print(f"Board bbox in image {imageId}:", board_bbox)
    print(f"Piece bboxes in image {imageId}:")
    # piece_bboxes holds bbox place
    for name, bboxes in piece_bboxes.items():
        print(f"  {name}: {bboxes}")
    return board_bbox, piece_bboxes

def getCameraExtrinsics(imageId): # will need to modularize later
    placements = load_json("/Users/rolandspoofy/developer/chess-datagen/gen-data/data_map/annotations/board_placements.json")
    img_key = f"images/{imageId}.png"
    cam = placements[img_key]["cam"]
    camera_position = np.array(cam["pos"])
    rot = np.array(cam["rot"]) # rotation matrix (3x3)
    origin_r = np.linalg.inv(rot)
    R_x_180 = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
    origin_r = R_x_180 @ origin_r

    # R = gt.look_at_rotation(T).T
    R = origin_r
    return camera_position, R

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


def dict_to_fen_placement(piece_map):

    board = [['' for _ in range(8)] for _ in range(8)]
    
    for piece, squares in piece_map.items():
        for sq in squares:
            file = ord(sq[0]) - ord('a')   
            rank = int(sq[1])               
            if 0 <= file < 8 and 1 <= rank <= 8:
                board[rank-1][file] = piece
    
    rows = []
    for r in range(7, -1, -1):
        empty = 0
        fen_row = ''
        for f in range(8):
            c = board[r][f]
            if c == '':
                empty += 1
            else:
                if empty:
                    fen_row += str(empty)
                    empty = 0
                fen_row += c
        if empty:
            fen_row += str(empty)
        rows.append(fen_row)
    
    return '/'.join(rows)


def piece_to_square(piece_bboxes,M):
    square_size = 256
    cell_size   = square_size / 8.0
    files       = "abcdefgh"
    
    board = [['' for _ in range(8)] for _ in range(8)]
    
    
    def name_to_fen(piece_name):
        name = piece_name.lower()
        if 'white' in name:
            color = 'white'
        elif 'black' in name:
            color = 'black'
        
        for ptype in ('pawn','knight','bishop','rook','queen','king'):
            if ptype in name:
                break

        base = {
            'pawn':   'P',
            'knight': 'N',
            'bishop': 'B',
            'rook':   'R',
            'queen':  'Q',
            'king':   'K'
        }[ptype]
        return base if color == 'white' else base.lower()
    
    #print("------------\n", piece_bboxes.items())
    piece_to_fen = defaultdict(list)
    for piece_name, bboxes in piece_bboxes.items():
        code = name_to_fen(piece_name)
        for bbox in bboxes:
            # get the point in the original image
            x, y, w, h = bbox
            cx, cy = x + (2*w)//4.0, y + (3.99*h)//4.0
            # warp into our persepctive
            src_pt = np.array([[[cx, cy]]], dtype=np.float32)       
            dst_pt = cv2.perspectiveTransform(src_pt, M)[0,0]       
            u, v   = dst_pt

            # find file (column) and rank (row) indices
            # 0...7 -> a..h
            # 0...7, 0 -> top row
            # 0 -> bottom (rank 1)
            file_idx       = int(u // cell_size)                     
            rank_from_top  = int(v // cell_size)                     
            rank_idx       = 7 - rank_from_top                       

            # clamp to [0,7]
            file_idx = max(0, min(7, file_idx))
            rank_idx = max(0, min(7, rank_idx))

            # make square name
            square = f"{files[file_idx]}{rank_idx+1}"
            piece_to_fen[code].append(square)
    #print("----")
    #print(piece_to_fen)
    return dict_to_fen_placement(piece_to_fen)


def main():
    imgNum = 7

    image = cv2.imread(f"gen-data/data_map/images/00000{imgNum}.png")

    board_bbox, piece_bboxes = getBbox(imgNum)

    camera_position, rot = getCameraExtrinsics(f"00000{imgNum}")

    R_c2w = look_at_rotation(camera_position)
    R, T = rot, camera_position
    K = np.array([
        [888.88909234, 0, 319.5],
        [0, 888.88909234, 239.5],
        [0, 0, 1]
    ])

    x_l, y_l, z_l = 2, 2, 0

    projected_pts = get_projected_rectangle_corners(x_l, y_l, z_l, R, T, K)

    print("Projected points:", projected_pts)

    output_img = image.copy()
    for p in projected_pts:
        cv2.circle(output_img, (int(round(p[0])), int(round(p[1]))), 8, (255, 0, 0), -1)

    for name, bboxes in piece_bboxes.items():
        for x,y,cx,cy in bboxes:
            cv2.circle(output_img, (int(x + (cx*2)//4) , int(y + (cy*3.99)//4)), 6, (50, 255, 177), -1)

    origin_world = np.array([[0, 0, 0]])
    origin_cam = (R @ (origin_world - T).T).T
    origin_proj = (K @ origin_cam.T).T
    origin_img = origin_proj[:, :2] / origin_cam[:, 2][:, np.newaxis]
    origin_pt = origin_img[0]


    # warp board
    warped_square, M, dst = warp_to_square(output_img, projected_pts, output_size=256)


    # Color
    # colored_square = color_grid_on_square(warped_square, grid_size=8)
    # warp back to original
    # final_result = warp_square_back(output_img, colored_square, projected_pts, square_size=256)


    # cv2.imshow("Projected Corners", output_img)
    # cv2.imshow("Perspective Transform", warped_square)
    #cv2.imshow("Final Overlay", final_result)
    fen = piece_to_square(piece_bboxes,M)
    print(fen)
    return fen

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

main()