from ultralytics.data.converter import convert_coco

convert_coco("coco_data_2025_07_09__00_10_55/test/annotations/", "coco_converted/",cls91to80=False) # this only grabs the annotations; need to grab for each folder

# after grabbing annotations; copy in images in the respective train val test folders
# apply random backgrounds for each image
# look into inserting images using multiple random backgrounds and adding to annotatoins? might be too much work; generating 20k images anyways