import blenderproc as bproc  # On version 2.8.0

import csv
import random
import time
import re
import bpy  # type: ignore
import random
import re
import os
import numpy as np
import re
import json
import os
import math
import datetime
# Init BlenderProc and Optimize
bproc.init()

# Load your scene
loaded = bproc.loader.load_blend("ChessBoard.blend")
output_path = datetime.datetime.now().strftime("coco_data_%Y_%m_%d__%H_%M_%S")
os.makedirs(output_path, exist_ok=True)
# ----- PIECE PLACEMENT -----
# This section defines the positions of the chess pieces on the board.
positionsDict = {
    'A1': (-0.87574, -0.87857),
    'A2': (-0.87574, -0.628312),
    'A3': (-0.87574, -0.378054),
    'A4': (-0.87574, -0.127796),
    'A5': (-0.87574, 0.122462),
    'A6': (-0.87574, 0.372720),
    'A7': (-0.87574, 0.622978),
    'A8': (-0.87574, 0.873236),
    'B1': (-0.62914, -0.87857),
    'B2': (-0.62914, -0.628312),
    'B3': (-0.62914, -0.378054),
    'B4': (-0.62914, -0.127796),
    'B5': (-0.62914, 0.122462),
    'B6': (-0.62914, 0.372720),
    'B7': (-0.62914, 0.622978),
    'B8': (-0.62914, 0.873236),
    'C1': (-0.38254, -0.87857),
    'C2': (-0.38254, -0.628312),
    'C3': (-0.38254, -0.378054),
    'C4': (-0.38254, -0.127796),
    'C5': (-0.38254, 0.122462),
    'C6': (-0.38254, 0.372720),
    'C7': (-0.38254, 0.622978),
    'C8': (-0.38254, 0.873236),
    'D1': (-0.13594, -0.87857),
    'D2': (-0.13594, -0.628312),
    'D3': (-0.13594, -0.378054),
    'D4': (-0.13594, -0.127796),
    'D5': (-0.13594, 0.122462),
    'D6': (-0.13594, 0.372720),
    'D7': (-0.13594, 0.622978),
    'D8': (-0.13594, 0.873236),
    'E1': (0.11066, -0.87857),
    'E2': (0.11066, -0.628312),
    'E3': (0.11066, -0.378054),
    'E4': (0.11066, -0.127796),
    'E5': (0.11066, 0.122462),
    'E6': (0.11066, 0.372720),
    'E7': (0.11066, 0.622978),
    'E8': (0.11066, 0.873236),
    'F1': (0.35726, -0.87857),
    'F2': (0.35726, -0.628312),
    'F3': (0.35726, -0.378054),
    'F4': (0.35726, -0.127796),
    'F5': (0.35726, 0.122462),
    'F6': (0.35726, 0.372720),
    'F7': (0.35726, 0.622978),
    'F8': (0.35726, 0.873236),
    'G1': (0.60386, -0.87857),
    'G2': (0.60386, -0.628312),
    'G3': (0.60386, -0.378054),
    'G4': (0.60386, -0.127796),
    'G5': (0.60386, 0.122462),
    'G6': (0.60386, 0.372720),
    'G7': (0.60386, 0.622978),
    'G8': (0.60386, 0.873236),
    'H1': (0.85046, -0.87857),
    'H2': (0.85046, -0.628312),
    'H3': (0.85046, -0.378054),
    'H4': (0.85046, -0.127796),
    'H5': (0.85046, 0.122462),
    'H6': (0.85046, 0.372720),
    'H7': (0.85046, 0.622978),
    'H8': (0.85046, 0.873236),
    'None' : (1,1)
}

pieceToSquareDict = {
    'BlackRook1':   'A8',
    'BlackRook2':   'H8',
    'BlackRook3':   'None',
    'BlackRook4':   'None',
    'BlackRook5':   'None',
    'BlackRook6':   'None',
    'BlackRook7':   'None',
    'BlackRook8':   'None',
    'BlackRook9':   'None',
    'BlackRook10':  'None',

    'BlackKnight1': 'B8',
    'BlackKnight2': 'G8',
    'BlackKnight3': 'None',
    'BlackKnight4': 'None',
    'BlackKnight5': 'None',
    'BlackKnight6': 'None',
    'BlackKnight7': 'None',
    'BlackKnight8': 'None',
    'BlackKnight9': 'None',
    'BlackKnight10':'None',

    'BlackBishop1': 'C8',
    'BlackBishop2': 'E8',
    'BlackBishop3': 'None',
    'BlackBishop4': 'None',
    'BlackBishop5': 'None',
    'BlackBishop6': 'None',
    'BlackBishop7': 'None',
    'BlackBishop8': 'None',
    'BlackBishop9': 'None',
    'BlackBishop10':'None',

    'BlackQueen1':  'D8',
    'BlackQueen2':  'None',
    'BlackQueen3':  'None',
    'BlackQueen4':  'None',
    'BlackQueen5':  'None',
    'BlackQueen6':  'None',
    'BlackQueen7':  'None',
    'BlackQueen8':  'None',
    'BlackQueen9':  'None',

    'BlackKing1':   'E8',

    'BlackPawn1':   'A7',
    'BlackPawn2':   'B7',
    'BlackPawn3':   'C7',
    'BlackPawn4':   'D7',
    'BlackPawn5':   'E7',
    'BlackPawn6':   'F7',
    'BlackPawn7':   'G7',
    'BlackPawn8':   'H7',

    'WhiteRook1':   'A1',
    'WhiteRook2':   'H1',
    'WhiteRook3':   'None',
    'WhiteRook4':   'None',
    'WhiteRook5':   'None',
    'WhiteRook6':   'None',
    'WhiteRook7':   'None',
    'WhiteRook8':   'None',
    'WhiteRook9':   'None',
    'WhiteRook10':  'None',

    'WhiteKnight1': 'B1',
    'WhiteKnight2': 'G1',
    'WhiteKnight3': 'None',
    'WhiteKnight4': 'None',
    'WhiteKnight5': 'None',
    'WhiteKnight6': 'None',
    'WhiteKnight7': 'None',
    'WhiteKnight8': 'None',
    'WhiteKnight9': 'None',
    'WhiteKnight10':'None',

    'WhiteBishop1': 'C1',
    'WhiteBishop2': 'F1',
    'WhiteBishop3': 'None',
    'WhiteBishop4': 'None',
    'WhiteBishop5': 'None',
    'WhiteBishop6': 'None',
    'WhiteBishop7': 'None',
    'WhiteBishop8': 'None',
    'WhiteBishop9': 'None',
    'WhiteBishop10':'None',

    'WhiteQueen1':  'D1',
    'WhiteQueen2':  'None',
    'WhiteQueen3':  'None',
    'WhiteQueen4':  'None',
    'WhiteQueen5':  'None',
    'WhiteQueen6':  'None',
    'WhiteQueen7':  'None',
    'WhiteQueen8':  'None',
    'WhiteQueen9':  'None',
    'WhiteQueen10': 'None',

    'WhiteKing1':   'E1',

    'WhitePawn1':   'A2',
    'WhitePawn2':   'B2',
    'WhitePawn3':   'C2',
    'WhitePawn4':   'D2',
    'WhitePawn5':   'E2',
    'WhitePawn6':   'F2',
    'WhitePawn7':   'G2',
    'WhitePawn8':   'H2',
}

def parse_fen(fen):
    files = 'abcdefgh'
    ranks = fen.split()[0].split('/')
    # print(ranks)
    piecePositions = {
        'BlackPawn': [], 'WhitePawn': [],
        'BlackRook': [], 'WhiteRook': [],
        'BlackKnight': [], 'WhiteKnight': [],
        'BlackBishop': [], 'WhiteBishop': [],
        'BlackQueen': [], 'WhiteQueen': [],
        'BlackKing': [],  'WhiteKing': [],
    }
    symbolToType = {
        'p': 'Pawn', 'r': 'Rook', 'n': 'Knight',
        'b': 'Bishop','q': 'Queen','k': 'King'
    }
    for rank_idx, rank in enumerate(ranks):
        file_idx = 0
        for ch in rank:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                color = 'White' if ch.isupper() else 'Black'
                ptype = symbolToType[ch.lower()]
                square = files[file_idx] + str(8 - rank_idx)
                piecePositions[f'{color}{ptype}'].append(square)
                file_idx += 1
    return piecePositions

# update my dict based on new positions
def update_dict_from_positions(d, positions):
    for k in d:
        d[k] = 'None'
    for ptype, squares in positions.items():
        for i, sq in enumerate(squares, start=1):
            key = f'{ptype}{i}'
            d[key] = sq
    return d

class Piece:
    def __init__(self,name, kind, color, initial_position, home_tile_color):
        self.name_= name
        self.kind_= kind
        self.color_= color
        self.initial_position_ = initial_position
        self.home_tile_color_ = home_tile_color

def allowed_squares_for(piece, free_squares):

    choices = set(free_squares)

    # pawns cant go on rank 1 or 8
    if piece.kind_ == 'Pawn':
        choices = {sq for sq in choices if sq[1] not in ('1','8')}

    # bishops only on same color they started on
    if piece.kind_ == 'Bishop':
        # piece.home_tile_color_ is a bool: False = Black True = White
        choices = {sq for sq in choices if tile_color[sq] == piece.home_tile_color_}


    return choices

def randomizePositions():
    free_squares = set(positionsDict.keys())
    assignment = {}

    pieces = []

    # if piece were chosen, add it to pieces otherwise make it disappear
    for p in pieceList:
        if random.random() > random.random() or 'King' in p.name_: # each piece has a 50% chance of selection, but kings are always selected
            pieces.append(p)
        else:
            bpy.data.objects[p.name_].hide_viewport = True
            bpy.data.objects[p.name_].hide_render = True
            
    #shuffe the pieces
    random.shuffle(pieces)

    for p in pieces:
        legal = allowed_squares_for(p, free_squares)
        if not legal:
            continue
            #there are no bugs, only features
            raise RuntimeError(f"No legal square left for {p.name_}!")

        square = random.choice(list(legal))
        free_squares -= {square}
        assignment[p.name_] = square


    return assignment

FILES = 'ABCDEFGH'
RANKS = '12345678'

# mapping squares to tile color
tile_color = {
    square: ((FILES.index(square[0]) + RANKS.index(square[1])) % 2 == 1)
    for square in positionsDict
}


pieceList = []

#make the piece class and populate pieceList
for piece, square in pieceToSquareDict.items():
    name = piece
    color = 'Black' if 'Black' in name else 'White'
    kind = re.sub(r'\d+$', '', piece[len(color):])
    initial_position = square
    home_tile_color = 'Black' if ((FILES.index(square[0]) + RANKS.index(square[1])) % 2 == 0) else 'White'
    pieceList.append(Piece(name,kind,color,initial_position,home_tile_color))

#reset the pieces
for name,square in pieceToSquareDict.items():
    x,y = positionsDict[square]
    bpy.data.objects[name].location.x  = x
    bpy.data.objects[name].location.y  = y
    bpy.data.objects[name].hide_viewport = False
    bpy.data.objects[name].hide_render = False

# Labeling for COCO annotations

CATEGORY_NAME_TO_ID = {
    "WhitePawn": 0,
    "WhiteRook": 1,
    "WhiteKnight": 2,
    "WhiteBishop": 3,
    "WhiteQueen": 4,
    "WhiteKing": 5,

    "BlackPawn": 6,
    "BlackRook": 7,
    "BlackKnight": 8,
    "BlackBishop": 9,
    "BlackQueen": 10,
    "BlackKing": 11,

    "Board": 12
}

def get_base_name(name):
    # Remove trailing digits: 'WhitePawn1' → 'WhitePawn'
    return re.sub(r'\d+$', '', name)

for obj in loaded:
    full_name = obj.get_name()
    
    base_name = get_base_name(full_name)

    if base_name not in CATEGORY_NAME_TO_ID:
        print(f"WARNING: Unknown category name '{base_name}' for object '{full_name}'")
        continue

    obj.set_cp("category_id", CATEGORY_NAME_TO_ID[base_name])

# ----- CAMERA SETUP -----
# This section sets up the camera to orbit around the chessboard.
# The camera will be positioned at a distance and look towards the center of the board.

# The below code was taken from the BlenderProc documentation
# Create a point light next to it
light = bproc.types.Light()
light.set_location([0.0, 0.0, 2.0]) # Light above the chessboard
light.set_energy(1000.0)
bproc.camera.set_resolution(640, 480)  # Set the resolution of the rendered images
bproc.renderer.set_output_format(enable_transparency=True)


# GPT Soup to create a fibonacci sphere for camera positions
# This will create a set of camera positions that are evenly distributed around the sphere
  # Distance from origin
N = 200    # Number of cameras
num_random_setup = 100  # Number of random setups to generate
print(f"Generating {num_random_setup} random setups with {N} camera positions each...")
# Golden angle in radians
golden_angle = np.pi * (3 - np.sqrt(5))

# Add Camera poses
for i in range(N):
    
    rho = random.uniform(4.5, 6.5)

    z = 1 - (i) / (N - 1)            # z from 1 to -1
    radius = np.sqrt(1 - z * z)          # radius at that z
    theta = golden_angle * i             # azimuthal angle

    x = np.cos(theta) * radius
    y = np.sin(theta) * radius

    pos = rho * np.array([x, y, z])      # scale to radius rho

    # Camera looks at origin
    forward_vec = -pos / np.linalg.norm(pos)
    rotation = bproc.camera.rotation_from_forward_vec(forward_vec)

    cam_pose = bproc.math.build_transformation_mat(pos.tolist(), rotation)
    bproc.camera.add_camera_pose(cam_pose)

# Render for each randomized position
trn_val_tst_split = [6,2,2]
gcd = math.gcd(math.gcd(trn_val_tst_split[0], trn_val_tst_split[1]), trn_val_tst_split[2])
trn_val_tst_split = [x // gcd for x in trn_val_tst_split]
split_map = {0: 'train', 1: 'val', 2: 'test'}

# Set up csv reading
csv_file = open('positions.csv', newline='')
reader = csv.reader(csv_file)
next(reader, None)        
fen_rows = iter(reader) 

for z in range(num_random_setup):
    print(f"==== Render step {z+1}/{num_random_setup}... ====")
    current_time = time.time()
    # Randomly shuffle the pieceList to create a new random setup
    # Determine which split this iteration belongs to (0=train, 1=val, 2=test)

    # Magic number stuff to figure out which split this is
    # This is a s̶i̶m̶p̶l̶e̶ bad way to split the data into train, validation, and test sets
    # Please don't fire me in the future or flame me for this i plead innocence
    split_idx = z % sum(trn_val_tst_split)
    if split_idx < trn_val_tst_split[0]:
        split_idx = 0
    elif split_idx < trn_val_tst_split[0] + trn_val_tst_split[1]:
        split_idx = 1
    else:
        split_idx = 2

    dir_pre = split_map[split_idx]

    try:
        row = next(fen_rows)
    except StopIteration:
        print("No more FEN rows—stopping early.")
        break

    # parse + update data based off stream
    fen = row[0]
    pos = parse_fen(fen)
    update_dict_from_positions(pieceToSquareDict, pos)
    placement = pieceToSquareDict.copy()

    for name, square in placement.items():
        x, y = positionsDict[square]
        bpy.data.objects[name].location.x = x
        bpy.data.objects[name].location.y = y

    # Render segmentation data and produce instance attribute maps
    seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])
    # Render the scene
    data = bproc.renderer.render()
    # Write data to coco file
    image_paths = bproc.writer.write_coco_annotations(
        f'{output_path}/{dir_pre}',
        instance_segmaps=seg_data["instance_segmaps"], # type: ignore
        instance_attribute_maps=seg_data["instance_attribute_maps"], # type: ignore
        colors=data["colors"], # type: ignore
        color_file_format="PNG",
        append_to_existing_output=True,  # <-- important!
    ) 
    # THE OUTPUT OF WRITE COCO WAS MODIFIED TO RETURN THE IMAGE PATHS; ENSURE THIS IS DONE IN FUTURE CODE REVISIONS

    input_json_path = f'{output_path}/{dir_pre}/board_placements.json'

    # Load existing data or create new list
    if os.path.exists(input_json_path):
        with open(input_json_path, "r") as f:
            try:
                data_list = json.load(f)
            except json.JSONDecodeError:
                data_list = {}
    else:
        data_list = {}

    # Append new entries
    for path in image_paths:
        if path in data_list.keys():
            print(f"WARNING: File path '{path}' is already in the board_placements.json -> overwriting entry.")
        data_list[path] = placement

    # Save back to file
    with open(input_json_path, "w") as f:
        json.dump(data_list, f)
    print(f"==== Render step {z+1} completed in {time.time() - current_time:.2f} seconds. ====")