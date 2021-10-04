# Dependencies
import numpy as np
import tifffile
import pandas as pd
import pylab
import matplotlib.pylab as plt
from flim_tools.io import read_asc
from flim_tools.image_processing import normalize
from scipy.signal import convolve


def rectangular_to_phasor_point(g, s):
    print("not implemented yet")

    angle = np.pi - np.arctan2(s, -g)
    magnitude = np.sqrt(g ** 2 + s ** 2)
    return
