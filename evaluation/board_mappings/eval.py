import sys
import os
import time
import json
import numpy as np
import cv2
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data_map")))
import use_board_GPT