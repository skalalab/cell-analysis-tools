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


def rectangular_to_phasor_lifetimes_array(g, s):
    """ 
    Takes an array(image) of g and s points and
    converts them to angle and magnitude phasor arrays

    Parameters
    ----------
        g  : float 
            array of g coordinates (x-coordinates)
        s  : float 
            array of s coordinates (y-coordinates)

    Returns
    -------
        PhasorArray object : float
            lifetime_angles_array - array of angles for each pixel
            lifetime_magnitudes_array - array of magnitudes for each pixel
    """
    PhasorLifetimesArray = coll.namedtuple("PhasorLifetimesArray", "angles magnitudes")

    # calculate angles and magnitudes for all points
    angles = np.pi - np.arctan2(s, -g)
    magnitudes = np.sqrt(g ** 2 + s ** 2)
    return PhasorLifetimesArray(angles=angles, magnitudes=magnitudes)
