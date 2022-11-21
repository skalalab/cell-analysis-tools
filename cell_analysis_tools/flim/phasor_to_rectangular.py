# Dependencies
import collections as coll

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pylab
import tifffile
from scipy.signal import convolve

from cell_analysis_tools.image_processing import normalize
from cell_analysis_tools.io import read_asc


def phasor_to_rectangular(angle, magnitude):
    """ Gets x and y position of phasor in rectangular coordinates

    Parameters
    ----------
        angle  : float 
            phasor angle in radians
        magnitude  : float 
            phasor magnitude
    Returns
    -------
        g  : float 
            x axis coordinate
        s  : float
            y axis coordinate
    """
    point_rect = coll.namedtuple("point_rect", "g s")
    g = magnitude * np.cos(angle)
    s = magnitude * np.sin(angle)
    return point_rect(g=g, s=s)
