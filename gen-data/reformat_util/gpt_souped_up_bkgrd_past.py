# Generated using GPT power 
# Attempt optimizing paste_images_on_background.py file

import os
import random
import argparse
from PIL import Image
from multiprocessing import Pool, cpu_count

# Global variables (shared for multiprocessing)
loaded_backgrounds = []
args = None

def load_backgrounds(paths):
    """Preload all background images into memory (RGBA)."""
    return [
        Image.open(path).convert("RGBA")
        for path in paths
    ]

def process_image(file_path):
    """Processes a single image: pastes it on a random background and saves result."""
    if not file_path.lower().endswith(tuple(args.types)):
        return

    try:
        img = Image.open(file_path).convert("RGBA")
        img_w, img_h = img.size

        background = random.choice(loaded_backgrounds).copy().resize([img_w, img_h])
        background.paste(img, mask=img)

        file_name = os.path.basename(file_path)

        if args.overwrite:
            background.save(file_path)
        else:
            if args.output == "output":
                output_dir = os.path.join(args.images, "output")
            else:
                output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)
            background.save(output_path)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

def main():
    global loaded_backgrounds, args

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", type=str, required=True,
                        help="Path to object images to paste.")
    parser.add_argument("-b", "--backgrounds", type=str, required=True,
                        help="Path to background images to paste on.")
    parser.add_argument("-t", "--types", default=('jpg', 'jpeg', 'png'),
                        type=str, nargs='+',
                        help="File types to consider. Default: jp[e]g, png.")
    parser.add_argument("-w", "--overwrite", action="store_true",
                        help="Overwrite original files. Default: False.")
    parser.add_argument("-o", "--output", default="output", type=str,
                        help="Output directory. Default: 'output'.")
    args = parser.parse_args()

    # Collect all image and background paths
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.images)
        for file in files
        if file.lower().endswith(tuple(args.types))
    ]
    background_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(args.backgrounds)
        for file in files
        if file.lower().endswith(tuple(args.types))
    ]

    if not background_paths:
        print("No background images found.")
        return

    print(f"Found {len(image_paths)} image(s) and {len(background_paths)} background(s).")

    # Load backgrounds into memory
    loaded_backgrounds = load_backgrounds(background_paths)

    # Process in parallel
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_image, image_paths)


if __name__ == "__main__":
    main()
