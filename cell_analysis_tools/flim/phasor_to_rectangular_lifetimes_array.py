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


def phasor_to_rectangular_lifetimes_array(angles, magnitudes):
    """ Takes in two arrays of angles and magnitudes
    and converts them to array of g and s points
    
    Parameters
    ----------
        f : int 
            laser angular frequency
        angles : float 
            array of angles
        mangitudes : float
            array of magnitudes
    Returns
    -------
        RectangularLifetimesArray : Phasor collection object
            object with g and s array of lifetimes
                g - array of g-coordinates(x-coordinates)
                s - array of s-coordinates (y-coordinates)
    """
    RectangularLifetimesArray = coll.namedtuple("RectangularLifetimesArray", "g s")

    # calculate g and s
    g = magnitudes * np.cos(angles)
    s = magnitudes * np.sin(angles)

    return RectangularLifetimesArray(g=g, s=s)
