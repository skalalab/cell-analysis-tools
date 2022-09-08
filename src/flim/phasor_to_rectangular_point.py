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


def phasor_to_rectangular_point(angle, magnitude):
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
    Point = coll.namedtuple("Point", "g s")
    g = magnitude * np.cos(angle)
    s = magnitude * np.sin(angle)
    return Point(g=g, s=s)
