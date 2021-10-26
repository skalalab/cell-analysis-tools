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
