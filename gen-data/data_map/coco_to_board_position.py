import json
import argparse
import collections

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def find_image(coco, image_id=None, image_file=None):
    for img in coco['images']:
        if image_id is not None and img['id'] == image_id:
            return img
    raise ValueError(f"Image cannot be found: id={image_id}, file={image_file})")

def map_pieces_to_squares(coco, img):
    # map IDs -> piece names
    cat_id_to_name = {c['id']: c['name'] for c in coco['categories']}
    # fill board_id with board and pieces

    board_id = 0
    for category in coco['categories']:
        category_name = category['name'].lower()
        if category_name == 'board':
            board_id = category['id']
            break
    
    if not board_id:
        raise ValueError(" Cant find 'board' in categories")

    # find board binding boxes
    b_ann = None 

    for ann in coco['annotations']:
        same_image = ann['image_id'] == img['id']
        same_board = ann['category_id'] == board_id

        if same_image and same_board:
            b_ann = ann
            break 

    if b_ann is None:
        raise ValueError(f" Cant find board annotations for image id: {img['id']}")
    bx, by, bw, bh = b_ann['bbox']

    # Compute size of one square
    sq_w = bw / 8.0
    sq_h = bh / 8.0

    # map all chess pieces (skip the board)
    mappings = collections.defaultdict(list)
    for ann in coco['annotations']:
        if ann['image_id'] != img['id'] or ann['category_id'] == board_id:
            continue
        name = cat_id_to_name[ann['category_id']]
        x, y, w, h = ann['bbox']
        cx = x + w/2
        cy = y + h/2

        # find col and row
        col = int((cx - bx) / sq_w)
        row = int((cy - by) / sq_h)
        # clamp to 0 <-> 7
        col = max(0, min(7, col))
        row = max(0, min(7, row))

        # used ascii black magic from csci 127 
        file_letter = chr(ord('a') + col)
        # row=0 -> rank 8, row=7 -> rank 1 
        # bc FEN is in a "weird" format, its flipped 
        rank_number = 8 - row  
        square = f"{file_letter}{rank_number}"

        mappings[square].append(name)

    return mappings

def main():
    # map pieces from COCO to a1 -> h8 squares
    p = argparse.ArgumentParser()
    #path to coco_annotations.json
    p.add_argument('coco_file')
    # 
    
    group = p.add_mutually_exclusive_group(required=True)
    # image id
    group.add_argument('--image_id',   type=int)
    # image file name
    group.add_argument('--image_file', type=str)
    args = p.parse_args()

    coco = load_json(args.coco_file)


    img = find_image(coco, image_id=args.image_id, image_file=args.image_file)
    mapping = map_pieces_to_squares(coco, img)

    print(f"Map of {img['file_name']} (id={img['id']}):")
    for sq in sorted(mapping):
        print(f"  {sq}: {', '.join(mapping[sq])}")


main()
