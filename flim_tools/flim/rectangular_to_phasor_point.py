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


def rectangular_to_phasor_point(g, s):
    print("not implemented yet")

    angle = np.pi - np.arctan2(s, -g)
    magnitude = np.sqrt(g ** 2 + s ** 2)
    return
