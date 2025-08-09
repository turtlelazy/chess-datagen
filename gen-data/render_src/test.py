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
import argparse
# Init BlenderProc and Optimize Your Settings
# Using magical numbers found online
# https://blenderartists.org/t/options-to-speed-up-a-render/1515328
bproc.init()
bproc.renderer.set_cpu_threads(0)
# Check available denoisers


bproc.renderer.set_render_devices(use_only_cpu=False)




bproc.renderer.set_noise_threshold(0.1)

# seems to not go any faster after lowering from 32
bproc.renderer.set_max_amount_of_samples(32)

bproc.renderer.set_light_bounces(3,3,4,4,12,8,0)

# Load your scene
loaded = bproc.loader.load_blend("ChessBoard2.blend")