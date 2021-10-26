# Dependencies
import numpy as np
import tifffile
import pandas as pd
import collections as coll
import pylab
import matplotlib.pylab as plt
import collections as coll
from flim_tools.io import read_asc
from flim_tools.image_processing import normalize
from scipy.signal import convolve


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
