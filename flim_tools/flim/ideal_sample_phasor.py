
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

def ideal_sample_phasor(f, lifetime):
    '''
    Generates a phasor for a specific lifetime at a given frequency.

    Parameters
    ----------
        f : int
            laser rep rate
        lifetime : float
            lifetime of desired single exponential sample

    Returns
    -------
        angle : float
            angle of ideal sample
        magnitude : float
            magnitude of ideal sample
    '''

    Phasor = coll.namedtuple('Phasor', 'angle magnitude')

    # lifetime = lifetime * 1e-12  # lifetime in ns
    w = 2 * np.pi * f
    ### simulated point values
    ideal_g = 1/(1+(w**2 * lifetime**2))
    ideal_s = (w*lifetime)/(1+(w*lifetime)**2)
    
    #angle = np.arctan(ideal_s/ideal_g) # radians
    angle = np.pi-np.arctan2(ideal_s, -ideal_g) #in radians
    magnitude = np.sqrt(ideal_g**2 + ideal_s**2)

    print("Input lifetime: ", (1/w * ideal_s/ideal_g))
    return Phasor(angle=angle, magnitude=magnitude)
