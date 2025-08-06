import os
import json
import numpy as np
import cv2
from collections import defaultdict
import sys
import time
# ========== Utility Functions ==========

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_image_id_from_path(image_path):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    return int(filename)

# ========== Camera Functions ==========

def look_at_rotation(camera_position, target=np.array([0, 0, 0]), up=np.array([0, 0, -1])):
    z_axis = (target - camera_position)
    z_axis /= np.linalg.norm(z_axis)

    up_proj = up - np.dot(up, z_axis) * z_axis
    up_proj /= np.linalg.norm(up_proj)

    x_axis = np.cross(up_proj, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    R_c2w = np.stack([x_axis, y_axis, z_axis], axis=1)
    return R_c2w

def get_camera_extrinsics(image_id, placement_json):
    img_key = f"images/{str(image_id).zfill(6)}.png"
    cam = placement_json[img_key]["cam"]
    camera_position = np.array(cam["pos"])
    rot = np.array(cam["rot"])

    origin_r = np.linalg.inv(rot)
    R_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    origin_r = R_x_180 @ origin_r

    return camera_position, origin_r

# ========== Annotation Processing ==========

def get_bboxes(image_id, annotation_json):
    category_map = {cat["id"]: cat["name"] for cat in annotation_json["categories"]}
    board_cat_id = next((cid for cid, name in category_map.items() if name.lower() == "board"), None)
    if board_cat_id is None:
        raise ValueError("No category named 'Board' found.")

    piece_bboxes = defaultdict(list)
    board_bbox = None

    for ann in annotation_json["annotations"]:
        if ann["image_id"] != image_id:
            continue
        if ann["category_id"] == board_cat_id:
            board_bbox = ann["bbox"]
        else:
            piece_name = category_map[ann["category_id"]]
            piece_bboxes[piece_name].append(ann["bbox"])

    return board_bbox, piece_bboxes

# ========== Projection & Warping ==========

def get_projected_rectangle_corners(x_l, y_l, z_l, R, T, K):
    corners_world = np.array([
        [-x_l / 2, -y_l / 2, z_l / 2],
        [ x_l / 2, -y_l / 2, z_l / 2],
        [ x_l / 2,  y_l / 2, z_l / 2],
        [-x_l / 2,  y_l / 2, z_l / 2],
    ])
    corners_cam = (R @ (corners_world - T).T).T
    corners_proj = (K @ corners_cam.T).T
    corners_img = corners_proj[:, :2] / corners_cam[:, 2][:, np.newaxis]
    return corners_img.tolist()


def get_M_from_corners(pts, output_size=256):
    pts = np.array(pts, dtype="float32")
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return M

def warp_to_square(image, pts, output_size=256):
    pts = np.array(pts, dtype="float32")
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))
    return warped, M, dst

# ========== FEN Conversion ==========
def extract_piece_layout(fen_string):
    """
    Given a full FEN string, return only the piece layout portion.
    """
    return fen_string.strip().split()[0]

def dict_to_fen_placement(piece_map):
    board = [['' for _ in range(8)] for _ in range(8)]
    for piece, squares in piece_map.items():
        for sq in squares:
            file = ord(sq[0]) - ord('a')
            rank = int(sq[1])
            board[8-rank][file] = piece


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

def piece_to_square(piece_bboxes, M):
    square_size = 256
    cell_size = square_size / 8.0
    files = "abcdefgh"
    piece_to_fen = defaultdict(list)

    def name_to_fen(piece_name):
        name = piece_name.lower()
        color = 'white' if 'white' in name else 'black'
        for ptype in ('pawn', 'knight', 'bishop', 'rook', 'queen', 'king'):
            if ptype in name:
                break
        base = {'pawn': 'P', 'knight': 'N', 'bishop': 'B', 'rook': 'R', 'queen': 'Q', 'king': 'K'}[ptype]
        return base if color == 'white' else base.lower()

    for piece_name, bboxes in piece_bboxes.items():
        code = name_to_fen(piece_name)
        for bbox in bboxes:
            x, y, w, h = bbox
            cx, cy = x + (2*w)/4.0, y + (3.99*h)/4.0
            src_pt = np.array([[[cx, cy]]], dtype=np.float32)
            dst_pt = cv2.perspectiveTransform(src_pt, M)[0, 0]
            u, v = dst_pt
            file_idx = int(u // cell_size)
            rank_from_top = int(v // cell_size)
            rank_idx = 7 - rank_from_top
            file_idx = max(0, min(7, file_idx))
            rank_idx = max(0, min(7, rank_idx))
            square = f"{files[file_idx]}{rank_idx+1}"
            piece_to_fen[code].append(square)
    return dict_to_fen_placement(piece_to_fen)

def rotate_fen_piece_layout(fen_layout):
    def fen_to_matrix(fen):
        board = []
        for row in fen.split('/'):
            expanded = []
            for c in row:
                if c.isdigit():
                    expanded.extend(['.'] * int(c))
                else:
                    expanded.append(c)
            board.append(expanded)
        return board

    def matrix_to_fen(matrix):
        fen_rows = []
        for row in matrix:
            count = 0
            fen_row = ''
            for cell in row:
                if cell == '.':
                    count += 1
                else:
                    if count > 0:
                        fen_row += str(count)
                        count = 0
                    fen_row += cell
            if count > 0:
                fen_row += str(count)
            fen_rows.append(fen_row)
        return '/'.join(fen_rows)

    def rotate_90_clockwise(matrix):
        return [list(reversed(col)) for col in zip(*matrix)]

    def rotate_180(matrix):
        return [row[::-1] for row in matrix[::-1]]

    def rotate_270_clockwise(matrix):
        return [list(col) for col in zip(*matrix[::-1])][::-1]

    original = fen_to_matrix(fen_layout)
    rot_90 = rotate_90_clockwise(original)
    rot_180 = rotate_180(original)
    rot_270 = rotate_270_clockwise(original)

    return {
        '0°': matrix_to_fen(original),
        '90°': matrix_to_fen(rot_90),
        '180°': matrix_to_fen(rot_180),
        '270°': matrix_to_fen(rot_270)
    }

# ========== Main Processing Function ==========
def process_image_no_vis(image_id, annotation_json, placement_json):
    board_bbox, piece_bboxes = get_bboxes(image_id, annotation_json)
    camera_position, R = get_camera_extrinsics(image_id, placement_json)
    
    K = np.array([
        [888.88909234, 0, 319.5],
        [0, 888.88909234, 239.5],
        [0, 0, 1]
    ])
    x_l, y_l, z_l = 2, 2, 0
    projected_pts = get_projected_rectangle_corners(x_l, y_l, z_l, R, camera_position, K)

    M = get_M_from_corners(projected_pts)
    fen = piece_to_square(piece_bboxes, M)

    return fen

def process_image(image_path, annotation_path, placement_path):
    image_id = get_image_id_from_path(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    annotation_json = load_json(annotation_path)
    placement_json = load_json(placement_path)

    board_bbox, piece_bboxes = get_bboxes(image_id, annotation_json)
    camera_position, R = get_camera_extrinsics(image_id, placement_json)

    K = np.array([
        [888.88909234, 0, 319.5],
        [0, 888.88909234, 239.5],
        [0, 0, 1]
    ])

    x_l, y_l, z_l = 2, 2, 0
    projected_pts = get_projected_rectangle_corners(x_l, y_l, z_l, R, camera_position, K)

    warped_square, M, dst = warp_to_square(image, projected_pts, output_size=256)
    fen = piece_to_square(piece_bboxes, M)

    return fen, warped_square

# ========== Run Example ==========

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python use_board_GPT.py <image_path> <annotation_path> <placement_path>")
        sys.exit(1)
    # image_path = "gen-data/data_map/images/000007.png"
    # annotation_path = "gen-data/data_map/annotations/coco_annotations.json"
    # placement_path = "gen-data/data_map/annotations/board_placements.json"

    

    image_path = sys.argv[1]
    annotation_path = sys.argv[2]
    placement_path = sys.argv[3]

    start = time.time()
    placement_json = load_json(placement_path)
    annotation_json = load_json(annotation_path)
    print(f"Time taken to load json: {time.time() - start} seconds")

    original_fen = placement_json[f"images/{str(get_image_id_from_path(image_path)).zfill(6)}.png"]["board"]
    original_fen = extract_piece_layout(original_fen)

    start = time.time()
    fen = process_image_no_vis(get_image_id_from_path(image_path), annotation_json, placement_json) # inp pre-loaded jsons
    all_rotations = rotate_fen_piece_layout(fen)
    print(f"Time taken to process image: {time.time() - start} seconds")

    print(all_rotations)
    print(f"Original FEN: {original_fen}")
    for angle, layout in all_rotations.items():
        if layout == original_fen:
            print(f"Match found at {angle}: {layout}")
            break
    