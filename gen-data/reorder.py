import blenderproc as bproc  # On version 2.8.0

import random
import time
import re
import bpy
import random
import re
import os
import numpy as np

# Init BlenderProc
bproc.init()

# Load your scene
loaded = bproc.loader.load_blend("ChessBoard.blend")

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
}

pieceToSquareDict = {
    'BlackRook1':   'A1',
    'BlackKnight1': 'B1',
    'BlackBishop1': 'C1',
    'BlackQueen1':  'D1',
    'BlackKing1':   'E1',
    'BlackBishop2': 'F1',
    'BlackKnight2': 'G1',
    'BlackRook2':   'H1',
    'BlackPawn1':   'A2',
    'BlackPawn2':   'B2',
    'BlackPawn3':   'C2',
    'BlackPawn4':   'D2',
    'BlackPawn5':   'E2',
    'BlackPawn6':   'F2',
    'BlackPawn7':   'G2',
    'BlackPawn8':   'H2',
    'WhiteRook1':   'A8',
    'WhiteKnight1': 'B8',
    'WhiteBishop1': 'C8',
    'WhiteQueen1':  'D8',
    'WhiteKing1':   'E8',
    'WhiteBishop2': 'F8',
    'WhiteKnight2': 'G8',
    'WhiteRook2':   'H8',
    'WhitePawn1':   'A7',
    'WhitePawn2':   'B7',
    'WhitePawn3':   'C7',
    'WhitePawn4':   'D7',
    'WhitePawn5':   'E7',
    'WhitePawn6':   'F7',
    'WhitePawn7':   'G7',
    'WhitePawn8':   'H7',
}
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
        if random.random() > random.random() or 'King' in p.name_:
            pieces.append(p)
        else:
            bpy.data.objects[p.name_].hide_viewport = True
            
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

placement = randomizePositions()

# placement is now a dict mapping (EX: 'BlackPawn3' → 'E5')
for name, square in placement.items():
    x,y = positionsDict[square]
    bpy.data.objects[name].location.x  = x
    bpy.data.objects[name].location.y  = y

# The below code was taken from the BlenderProc documentation
# Create a point light next to it
light = bproc.types.Light()
light.set_location([0.0, 0.0, 2.0])
light.set_energy(1000.0)

# Set the camera to be in front of the object

# DO ORBITING CAMERA WITH FOR LOOP MAGIC
bproc.camera.set_resolution(480, 640)
cam_pose1 = bproc.math.build_transformation_mat([0, -5, 0], [np.pi / 2, 0, 0])
bproc.camera.add_camera_pose(cam_pose1)
cam_pose2 = bproc.math.build_transformation_mat([0, 5, 0], [np.pi / 2, 0, np.pi])
bproc.camera.add_camera_pose(cam_pose2)

# Get segmentation masks for all objects
# Set some category ids for loaded objects
for j, obj in enumerate(loaded):
    obj.set_cp("category_id", j+1)

# Render segmentation data and produce instance attribute maps
seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])


# Render the scene
data = bproc.renderer.render()

# # Write the rendering into an hdf5 file
# bproc.writer.write_hdf5("output/", data)

# Write data to coco file
bproc.writer.write_coco_annotations('coco_data',
                                    instance_segmaps=seg_data["instance_segmaps"],
                                    instance_attribute_maps=seg_data["instance_attribute_maps"],
                                    colors=data["colors"],
                                    color_file_format="JPEG")
