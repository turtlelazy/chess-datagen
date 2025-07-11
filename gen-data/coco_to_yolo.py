# NO AI WHATSOEVER CHALLENGE - Ishraq

from ultralytics.data.converter import convert_coco
import shutil
import os
import json

input_path = "coco_data_2025_07_09__00_21_27"
destination = "coco_converted"
# convert_coco(f"{input_path}/test/annotations/",
# "coco_converted/",cls91to80=False)
# # this only grabs the annotations; need to grab for each folder

# after grabbing annotations; copy in images in the respective train val test folders
# apply random backgrounds for each image
# look into inserting images using multiple random backgrounds and adding to annotatoins? 
# might be too much work; generating 20k images anyways

split = ["test", "train", "val"]

# Make copy of the dir to modify for coco format
new_path = input_path + "_copy"
shutil.copytree(input_path, new_path)
for s in split:
    os.mkdir(f"{new_path}/{s}/annotations")
    annotations_path = f"{new_path}/{s}/annotations/{s}.json"
    shutil.move(
        f"{new_path}/{s}/coco_annotations.json",
        annotations_path
    )

    # Update the json file for YOLO DS formatting
    anno_file = open(annotations_path, 'r')
    anno_json = json.load(anno_file)
    anno_file.close()
    for i in range(len(anno_json["images"])):
        anno_json["images"][i] = anno_json["images"][i]["file_name"]\
            .replace("images/", "")
    anno_file = open(annotations_path, 'w')
    anno_file.write(json.dumps(anno_json))
    anno_file.close()

# for s in split:
#     # Create folder for coco annotations
#     convert_coco(
#         f"{input_path}/{s}/", "coco_converted/",
#         # use_segments=True,
#         cls91to80=False
#     )
    # # Copy Images Over
    # images_path = f"{input_path}/{s}/annotations/"
    # result_path = f"{destination}/images/{s}"
    # shutil.copy(images_path, destination)