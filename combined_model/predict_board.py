import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from ultralytics import YOLO
import chess
import chess.svg
from cairosvg import svg2png
import matplotlib.pyplot as plt

# Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data_map")))
import use_board_GPT

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../evaluation")))
from ChessBoardDetector.Sams_X_corner.FindChessboards import processSingleCustomWrapper
from collections import defaultdict

# Original piece map
abbrev_to_full = {
    'WP': 'White Pawn',
    'WR': 'White Rook',
    'WN': 'White Knight',
    'WB': 'White Bishop',
    'WQ': 'White Queen',
    'WK': 'White King',
    'BP': 'Black Pawn',
    'BR': 'Black Rook',
    'BN': 'Black Knight',
    'BB': 'Black Bishop',
    'BQ': 'Black Queen',
    'BK': 'Black King',
    'BRD': 'Board'  # if you want to keep this
}


# Check CUDA
cuda_available = torch.cuda.is_available()
print("CUDA is available" if cuda_available else "WARNING: Use GPU for best performance")

def convert_yolo_to_bbox_dict(yolo_result, class_names):
    """
    Converts YOLO results to dict: {piece_name: [bboxes]}.
    Each bbox = [x, y, w, h] with (x, y) as top-left corner.
    """

    piece_bboxes = defaultdict(list)
    for box in yolo_result.boxes:
        cls_id = int(box.cls.item())
        piece_name = abbrev_to_full[class_names[cls_id]]

        # box.xywh is center x, y, w, h
        x_c, y_c, w, h = box.xywh[0].cpu().numpy()
        x = x_c - w / 2
        y = y_c - h / 2

        piece_bboxes[piece_name].append([x, y, w, h])

    return piece_bboxes

# ------------------ Main Logic ------------------
if __name__ == "__main__":
    # Input
    example_image_path = "samples/000571.png"
    model_path = "models/train4/weights/best.pt"
    model = YOLO(model_path)

    # Load Image
    image = cv2.imread(example_image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {example_image_path}")
    
    image_id = use_board_GPT.get_image_id_from_path(example_image_path)

    # YOLO Inference
    results = model(example_image_path)
    boxes = results[0].boxes
    class_names = model.names

    # Parse YOLO into dict[category_name] = [bboxes]
    piece_bboxes = convert_yolo_to_bbox_dict(results[0], class_names)
    print(piece_bboxes)
    # Corner Detection
    predicted_pts = processSingleCustomWrapper(example_image_path)
    if predicted_pts is None or len(predicted_pts) != 4:
        raise ValueError("Board corners could not be detected properly.")

    # Warp image & get homography
    warped_square, M, dst_pts = use_board_GPT.warp_to_square(image, predicted_pts, output_size=256)

    # Predict FEN using piece bbox + homography
    fen_layout = use_board_GPT.piece_to_square(piece_bboxes, M)

    # Rotate FEN to canonical orientations
    rotated_fens = use_board_GPT.rotate_fen_piece_layout(fen_layout)

    # Output
    print("All Rotated FENs:")
    for angle, fen in rotated_fens.items():
        print(f"{angle}: {fen}")
    debug_image = warped_square.copy()
    square_size = 256
    cell_size = square_size / 8
    main_fen = rotated_fens["0°"]
    board = chess.Board(main_fen + " w - - 0 1")
    svg_data = chess.svg.board(board=board, size=400)
    svg2png(bytestring=svg_data.encode('utf-8'), write_to="board.png")

    corner_img = image.copy()
    if predicted_pts is not None and len(predicted_pts) == 4:
        for i, pt in enumerate(predicted_pts):
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(corner_img, (x, y), 8, (0, 0, 255), -1)  # red dots
            cv2.putText(corner_img, f"C{i+1}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- Draw YOLO bounding boxes with YOLO labels ---
    for box in results[0].boxes:
        cls_id = int(box.cls.item())
        label = class_names[cls_id]  # from YOLO model
        x_c, y_c, w, h = box.xywh[0].cpu().numpy()
        x1, y1 = int(x_c - w / 2), int(y_c - h / 2)
        x2, y2 = int(x_c + w / 2), int(y_c + h / 2)

        cv2.rectangle(corner_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(corner_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite("detections_and_corners.png", corner_img)
    
    # Load images for display
    # input_img = cv2.cvtColor(cv2.imread(example_image_path), cv2.COLOR_BGR2RGB)
    board_img = np.array(Image.open("board.png"))

    # Display side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(corner_img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(board_img)
    axes[1].set_title("Detected Board Position")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("side_by_side.png", dpi=300)


