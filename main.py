# Imports

import cv2
import numpy as np
import argparse

from test import *
from live import *

# This code here is so that you can test it with the images in the test folder

testDetection()

# This code here is designed to be used on the Raspberry Pi with a camera attached and led lights and
# pointing at a raw pancake/caramelised sugar in a pan (although I haven't changed the video input statement to recognise the pi camera
# in case there is the off chance you want to test this with a normal camera and raw pancakes)

# liveDetection()


