import numpy as np
import json
import get_board_GT as gt
import cv2
import matplotlib.pyplot as plt


def intersect_union_score(img1, img2):
    """
    Calculates the Intersection over Union (IoU) for each pixel color between two images.
    Args:
        img1: numpy array of shape (H, W, C) or (H, W)
        img2: numpy array of shape (H, W, C) or (H, W)
    Returns:
        dict mapping color (as tuple) to IoU score
    """
    # Ensure images are the same shape
    assert img1.shape == img2.shape, "Images must have the same shape"
    # Flatten images to (N, C) or (N,)
    flat1 = img1.reshape(-1, img1.shape[-1]) if img1.ndim == 3 else img1.flatten()
    flat2 = img2.reshape(-1, img2.shape[-1]) if img2.ndim == 3 else img2.flatten()
    # Get all unique colors in both images
    colors = set(map(tuple, np.unique(flat1, axis=0))) | set(map(tuple, np.unique(flat2, axis=0)))
    print(len(colors))
    iou_scores = {}
    # for color in colors:
    #     mask1 = np.all(flat1 == color, axis=-1) if img1.ndim == 3 else flat1 == color
    #     mask2 = np.all(flat2 == color, axis=-1) if img2.ndim == 3 else flat2 == color
    #     intersection = np.logical_and(mask1, mask2).sum()
    #     union = np.logical_or(mask1, mask2).sum()
    #     iou = intersection / union if union > 0 else 0.0
    #     iou_scores[color] = iou
    return iou_scores


def test_coco(dir, method, vis=False):
    board_json = open(f"{dir}/board_placements.json")
    board_json = json.load(board_json)
    i = 0
    for key, value in board_json.items():
        cur_img_dir = f"{dir}/{key}"
        T = value["cam"]["pos"]
        origin_r = np.linalg.inv(np.array(value["cam"]["rot"]))
        R_x_180 = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        origin_r = R_x_180 @ origin_r

        # R = gt.look_at_rotation(T).T
        R = origin_r

        # print(R,"\n", origin_r)
        # TODO: FIX MAGIC NUMBERS; NEED TO PARAM
        K = np.array([
            [888.88909234, 0, 319.5],
            [0, 888.88909234, 239.5],
            [0, 0, 1]
        ]) # TODO: Grab this from data after incor in gen
        radius = 2
        x_l, y_l, z_l = radius, radius, 0

        curr_image = cv2.imread(cur_img_dir)

        gt_pts = gt.get_projected_rectangle_corners(x_l, y_l, z_l, R, T, K)
        predicted_pts = method(cur_img_dir)
        if predicted_pts is None:
            predicted_pts = []

        gt_mask = gt.get_board_mask(gt_pts, curr_image)
        try:
            predicted_mask = gt.get_board_mask(predicted_pts, curr_image)
        except Exception as e:
            predicted_mask = np.zeros_like(gt_mask)

        iou_scores = intersect_union_score(gt_mask, predicted_mask)
        print(f"Image: {key}, IoU Scores: {iou_scores}")

        if vis:
            print(f"Image: {key}, GT Points: {gt_pts}, Predicted Points: {predicted_pts}")
            print(f"GT Mask shape: {gt_mask.shape}, Predicted Mask shape: {predicted_mask.shape}")
            for p in predicted_pts:
                cv2.circle(curr_image, (int(round(p[0])), int(round(p[1]))), 8, (255, 0, 0), -1)
            for p in gt_pts:
                cv2.circle(curr_image, (int(round(p[0])), int(round(p[1]))), 8, (0, 255, 0), -1)
        
            print("Showing GT and Predicted Masks")
            cv2.imshow("Original Image", curr_image)
            cv2.imshow("GT Mask", gt_mask)
            cv2.imshow("Predicted Mask", predicted_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    
if  __name__ == "__main__":
    # Temp imports for testing
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ChessBoardDetector.Sams_X_corner.FindChessboards import processSingleCustomWrapper

    dir = "coco_data_2025_07_18__01_55_08/train"
    test_coco(dir, processSingleCustomWrapper, vis=True)