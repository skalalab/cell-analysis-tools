
# Dependencies 
import numpy as np
import os
import tifffile
import pandas as pd
import collections as coll
import pylab
# import plotly.graph_objs as go
# from plotly.offline import plot
#import sdtfile as sdt
import matplotlib.pylab as plt
import zipfile
import collections as coll
from pprint import pprint
from flim_tools.io import read_asc
from flim_tools.image_processing import normalize
from scipy.signal import convolve

def phasor_to_rectangular_point(angle, magnitude):
    ''' Gets x and y position of phasor in rectangular coordinates

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
    '''
    Point = coll.namedtuple('Point', 'g s')
    g = magnitude * np.cos(angle)
    s = magnitude * np.sin(angle)
    return Point(g=g, s=s)