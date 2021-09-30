# Dependencies
import numpy as np
import os
import tifffile
import pandas as pd
import collections as coll
import pylab

# import plotly.graph_objs as go
# from plotly.offline import plot
# import sdtfile as sdt
import matplotlib.pylab as plt
import zipfile
import collections as coll
from pprint import pprint
from flim_tools.io import read_asc
from flim_tools.image_processing import normalize
from scipy.signal import convolve


def td_to_fd(f, timebins, counts):
    """ Time to frequency domain transformation

    Parameters
    ---------- 
        f  : int 
            laser repetition angular frequency
        timebins : ndarray
            numpy array of timebins
        counts : ndarray
            photon counts of the histogram()

    Returns
    -------
        angle  : float 
            angle in radians
        magnitude  : float 
            magnitude of phasor
    """
    w = 2 * np.pi * f  # f
    Phasor = coll.namedtuple("Phasor", "angle magnitude")  # nameddtuple
    # pylab.plot(timebins,counts)

    ## convert to phasor rectangular
    point_g = np.sum(counts * np.cos(w * timebins)) / np.sum(counts)
    point_s = np.sum(counts * np.sin(w * timebins)) / np.sum(counts)

    # https://software.intel.com/en-us/forums/archived-visual-fortran-read-only/topic/313067
    # 0.5*TWOPI-ATAN2(Y,-X)
    # angle = AMOD(ATAN2(y,x)+TWOPI,TWOPI)
    angle = np.pi - np.arctan2(point_s, -point_g)
    magnitude = np.sqrt(point_g ** 2 + point_s ** 2)

    return Phasor(angle=angle, magnitude=magnitude)
