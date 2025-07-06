import blenderproc as bproc  # On version 2.8.0

import random
import time
import re
import bpy
import random
import re
import os

# Init BlenderProc
bproc.init()

# Load your scene
loaded = bproc.loader.load_blend("ChessBoard.blend")
