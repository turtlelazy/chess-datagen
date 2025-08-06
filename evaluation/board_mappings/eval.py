import sys
import os
import time
import json
import numpy as np
import cv2
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # Install with: pip install tqdm
import csv
import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data_map")))
import use_board_GPT


def process_single_image(image_str, placement_json, annotation_json):
    image_id = use_board_GPT.get_image_id_from_path(image_str)
    original_fen = use_board_GPT.extract_piece_layout(placement_json[image_str]["board"])
    
    try:
        fen = use_board_GPT.process_image_no_vis(image_id, annotation_json, placement_json)
        all_rotations = use_board_GPT.rotate_fen_piece_layout(fen)
        for layout in all_rotations.values():
            if layout == original_fen:
                return True, fen, original_fen
    except Exception as e:
        print(f"[Error] Processing {image_str}: {e}")

    return False, fen, original_fen


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval.py <coco_path>")
        sys.exit(1)

    dir_path = sys.argv[1]
    print(f"Using directory: {dir_path}")
    annotation_path = os.path.join(dir_path, "coco_annotations.json")
    placement_path = os.path.join(dir_path, "board_placements.json")

    start = time.time()
    placement_json = use_board_GPT.load_json(placement_path)
    annotation_json = use_board_GPT.load_json(annotation_path)
    print(f"Time taken to load json: {time.time() - start:.2f} seconds")

    total_images = len(placement_json)
    total_passed = 0

    # Output CSV path
    csv_output_path = f"board_mapping_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Open CSV writer
    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'passed', 'output_fen', 'original_fen'])  # header row

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(process_single_image, image_str, placement_json, annotation_json): image_str
                for image_str in placement_json
            }

            for future in tqdm(as_completed(futures), total=total_images, desc="Evaluating"):
                image_str = futures[future]
                try:
                    result, fen, original_fen = future.result()
                    writer.writerow([image_str, result, fen, original_fen])
                    if result:
                        total_passed += 1
                except Exception as e:
                    writer.writerow([image_str, "ERROR", ""])
                    print(f"[Error] Exception during processing {image_str}: {e}")

    print(f"Total Images: {total_images}, Total Passed: {total_passed}, Accuracy: {total_passed / total_images * 100:.2f}%")
