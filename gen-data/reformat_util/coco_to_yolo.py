# NO AI WHATSOEVER CHALLENGE - Ishraq

from ultralytics.data.converter import convert_coco
import shutil
import os
import json
import sys
import argparse

# convert_coco(f"{input_path}/test/annotations/",
# "coco_converted/",cls91to80=False)
# # this only grabs the annotations; need to grab for each folder

# after grabbing annotations; 
# copy in images in the respective train val test folders
# apply random backgrounds for each image
# look into inserting images using 
# multiple random backgrounds and adding to annotatoins? 
# might be too much work; generating 20k images anyways


def logger(str, to_log, log_func=print):
    if to_log:
        log_func(str)


def convert_bproc_coco_2_yolo(
        input_dir,
        destination,
        split=["test", "train", "val"],
        log=True
):

    logger("Creating temp anno files", log)
    anno_path = "temp_annotations"
    if os.path.exists(anno_path):
        shutil.rmtree(anno_path)  # purge into oblivion
    os.makedirs(anno_path)

    # Loop over to format anno files
    for s in split:
        # Copy files
        old_anno_path = f"{input_dir}/{s}/coco_annotations.json"
        annotations_path = f"{anno_path}/{s}.json"
        shutil.copy(
            old_anno_path,
            annotations_path
        )

        # Update the json file for YOLO DS formatting
        anno_file = open(annotations_path, 'r')
        anno_json = json.load(anno_file)
        anno_file.close()

        for i in range(len(anno_json["images"])):
            anno_json["images"][i]["file_name"] = \
                anno_json["images"][i]["file_name"] \
                .replace("images/", "")
        anno_file = open(annotations_path, 'w')
        anno_file.write(json.dumps(anno_json))
        anno_file.close()

    # Get annotations in YOLO format
    logger("Getting YOLO labels", log)
    copy_count = 0
    base = f"{destination}"
    store_dir = base
    while os.path.exists(store_dir):
        copy_count += 1
        store_dir = base + str(copy_count)

    convert_coco(
        anno_path, store_dir,
        # use_segments=True, # DO NOT USE; doesnt work for some reason
        cls91to80=False
    )

    # Copy images
    logger("Copying images into YOLO folder", log)
    for s in split:
        img_path = f"{store_dir}/images/{s}"
        shutil.copytree(f"{input_dir}/{s}/images", img_path)

    logger("Deleting temp dir", log)
    # Remove temp dir
    shutil.rmtree(anno_path)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        parser = argparse.ArgumentParser(
            description="Convert COCO to YOLO format."
        )
        parser.add_argument(
            '--input',
            type=str, required=True,
            help='Input directory'
        )
        parser.add_argument(
            '--dest',
            type=str,
            required=True,
            help='Destination directory'
        )
        parser.add_argument(
            '--verbose',
            type=str,
            required=False,
            help='Destination directory'
        )
        args = parser.parse_args()
        input_dir = args.input
        destination = args.dest
        log = True
    else:
        input_dir = "coco_data_2025_07_09__00_21_27"
        destination = "coco_converted"
        log = False

    convert_bproc_coco_2_yolo(
        input_dir=input_dir,
        destination=destination,
        log=log
    )