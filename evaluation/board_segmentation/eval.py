import numpy as np
import json
import get_board_GT as gt
import cv2
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import datetime

def reorder_points(predicted_pts, gt_pts):
    """
    Reorders predicted_pts to best match gt_pts using the Hungarian algorithm.
    
    Args:
        predicted_pts: (N, 2) or (N, D) numpy array of predicted points.
        gt_pts: (N, 2) or (N, D) numpy array of ground truth points.
        
    Returns:
        reordered_predicted_pts: predicted_pts reordered to best match gt_pts.
        indices: index mapping from gt_pts to predicted_pts
    """
    # Compute pairwise distance matrix
    cost_matrix = np.linalg.norm(gt_pts[:, None, :] - predicted_pts[None, :, :], axis=-1)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Return reordered predicted points
    reordered_predicted_pts = predicted_pts[col_ind]
    return reordered_predicted_pts, col_ind


def intersect_union_score(img1, img2):
    """
    Vectorized IoU computation for per-pixel color between two images.
    """
    assert img1.shape == img2.shape, "Images must have the same shape"

    # Flatten to (N, C) or (N,)
    flat1 = img1.reshape(-1, img1.shape[-1]) if img1.ndim == 3 else img1.flatten()[:, None]
    flat2 = img2.reshape(-1, img2.shape[-1]) if img2.ndim == 3 else img2.flatten()[:, None]

    # Concatenate all pixels to compute consistent labels
    all_colors = np.vstack([flat1, flat2])

    # Assign integer labels to unique colors
    unique_colors, labels = np.unique(all_colors, axis=0, return_inverse=True)
    labels1 = labels[:len(flat1)]
    labels2 = labels[len(flat1):]

    # Number of color classes
    num_colors = len(unique_colors)

    # Initialize IoU dict
    iou_scores = {}

    for i in range(num_colors):
        mask1 = labels1 == i
        mask2 = labels2 == i
        intersection = np.sum(mask1 & mask2)
        union = np.sum(mask1 | mask2)
        color_tuple = tuple(unique_colors[i]) if img1.ndim == 3 else unique_colors[i][0]
        iou_scores[color_tuple] = intersection / union if union > 0 else 0.0

    return iou_scores

def iou_to_score(iou_scores):
    """
    Converts IoU scores to a single score.
    Args:
        iou_scores: dict mapping color (as tuple) to IoU score
    Returns:
        float: average IoU score across all colors
    """
    if not iou_scores:
        return 0.0
    return sum(iou_scores.values()) / len(iou_scores)

def test_coco(dir, method, metric=iou_to_score, vis=False, backg = True):
    results = {}

    board_json = open(f"{dir}/board_placements.json")
    board_json = json.load(board_json)
    i = 0

    output_file = f"board_segmentation_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(output_file, 'a') as fw:
        fw.write("Image,Score\n")

    for key, value in board_json.items():
        i += 1
        start_start = time.time()
        # if key != "images/000156.png":
        #     continue
        try:

            print(f"Processing Image: {key}")

            cur_img_dir = f"{dir}/{key}"
            if backg:
                cur_img_dir = f"{dir}/images/output/{key.replace("images/", "")}"

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
            start = time.time()
            gt_pts = gt.get_projected_rectangle_corners(x_l, y_l, z_l, R, T, K)
            print(f"\tGT Retrieval: {time.time() - start:.4f} seconds")

            start = time.time()
            predicted_pts = method(cur_img_dir)

            predicted_pts = np.array(predicted_pts)
            gt_pts = np.array(gt_pts)

            print(f"\tPredicted Retrieval: {time.time() - start:.4f} seconds")
            predicted_pts, col = reorder_points(predicted_pts, gt_pts)
            # rotated_predicted_pts = []
            # if predicted_pts is None:
            #     predicted_pts = []
            # else:
            #     rotated_predicted_pts = [np.roll(predicted_pts, -i, axis=0) for i in range(len(predicted_pts))]

            best_predicted_pts = None
            gt_mask = gt.get_board_mask(gt_pts, curr_image)

            try:
                curr_predicted_mask = gt.get_board_mask(predicted_pts, curr_image)
            except Exception as e:
                curr_predicted_mask = np.zeros_like(gt_mask)
            iou_scores = intersect_union_score(gt_mask, curr_predicted_mask)
            score = metric(iou_scores)
            results[key] = score
            print(f"\tImage: {key}, Score: {score}")
            print()
            if vis:
                print(f"Image: {key}, GT Points: {gt_pts}, Predicted Points: {predicted_pts}")
                print(
                    f"GT Mask shape: {gt_mask.shape}, Predicted Mask shape: {curr_predicted_mask.shape}"
                )
                for p in predicted_pts:
                    cv2.circle(curr_image, (int(round(p[0])), int(round(p[1]))), 8, (255, 0, 0), -1)
                for p in gt_pts:
                    cv2.circle(curr_image, (int(round(p[0])), int(round(p[1]))), 8, (0, 255, 0), -1)

                print("Showing GT and Predicted Masks")
                cv2.imshow("Original Image", curr_image)
                cv2.imshow("GT Mask", gt_mask)
                cv2.imshow("Predicted Mask", curr_predicted_mask)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error processing image {key}: {e}")
            continue
        with open(output_file, 'a') as fw:
            fw.write(f"{key},{score}\n")
        print(f"Time taken for image {i}: {time.time() - start_start:.4f} seconds")
    output_file.close()
if  __name__ == "__main__":
    # Temp imports for testing
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from ChessBoardDetector.Sams_X_corner.FindChessboards import processSingleCustomWrapper

    # dir = "coco_data_2025_07_18__01_55_08/train"
    dir = "../gen-data/render_src/coco_data_2025_07_22__23_40_11/train"
    test_coco(dir, processSingleCustomWrapper, vis=True, backg=True)
