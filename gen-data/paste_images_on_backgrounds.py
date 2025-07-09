# Note my code; Source: https://github.com/DLR-RM/BlenderProc/blob/main/examples/advanced/random_backgrounds/paste_images_on_backgrounds.py

"""This script allows to automatically paste generated images on random backgrounds."""

import os
import random
import argparse
from PIL import Image


def main():
    # Get and parse all given arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",
                        "--images",
                        type=str,
                        help="Path to object images to paste.")
    parser.add_argument("-b",
                        "--backgrounds",
                        type=str,
                        help="Path to background images to paste on.")
    parser.add_argument("-t",
                        "--types",
                        default=('jpg', 'jpeg', 'png'),
                        type=str,
                        nargs='+',
                        help="File types to consider. Default: jp[e]g, png.")
    parser.add_argument(
        "-w",
        "--overwrite",
        action="store_true",
        help=
        "Merges images and backgrounds, overwriting original files. Default: False."
    )
    parser.add_argument("-o",
                        "--output",
                        default="output",
                        type=str,
                        help="Output directory. Default: 'output'.")
    args = parser.parse_args()

    # Create an output directory if `overwrite` is not selected
    if not args.overwrite:
        if args.output == "output":
            os.makedirs(os.path.join(args.images, "output"), exist_ok=True)
        else:
            os.makedirs(args.output, exist_ok=True)
    random_backgrounds = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.backgrounds)
        for file in files
        if file.lower().endswith(tuple(args.types))
    ]
    print(random_backgrounds)
    # Go through all files in given `images` directory
    for file_name in os.listdir(args.images):
        # Matching files to given `types` and opening images
        # print(file_name)
        if file_name.lower().endswith(args.types):
            img_path = os.path.join(args.images, file_name)
            img = Image.open(img_path)
            img_w, img_h = img.size

            # Selecting and opening a random image file from given `backgrounds` directory to use as background
            background_path = random.choice(random_backgrounds)
            print("bkg path", background_path)
            background = Image.open(background_path).resize([img_w, img_h])
            # Pasting the current image on the selected background
            background.paste(img, mask=img.convert('RGBA'))

            # Overwrites original image with merged one if `overwrite` is selected
            if args.overwrite:
                background.save(img_path)
            # Else store merged image in default or provided `output` directory
            else:
                if args.output == "output":
                    background.save(
                        os.path.join(args.images, "output", file_name))
                else:
                    background.save(args.output)


if __name__ == "__main__":
    main()