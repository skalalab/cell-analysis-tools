# Dependencies
import collections as coll

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pylab
import tifffile
from scipy.signal import convolve

from cell_analysis_tools.flim import ideal_sample_phasor, td_to_fd
from cell_analysis_tools.image_processing import normalize
from cell_analysis_tools.io import read_asc


def phasor_calibration(f, lifetime, timebins, counts):
    """ Phasor Plot Calibration

    Parameters
    ---------- 
        f  : int 
            laser repetition angular frequency
        lifetime  : int 
            lifetime of known sample in ns (single exponential decay)
        timebins  : int 
            timebins of samples
        counts  : int 
            photon counts of histogram
    Returns
    -------
        angle_offset {float}: difference in angle between known sample and actual
        magnitude_offset {float}: difference in magnitude between known sample and actual
    Note
    ----
        If no timebins or histograms passed then returns angle and phase of
        lifetime passed in
    """
    Calibration = coll.namedtuple("Calibration", "angle scaling_factor")

    # calculate idea and real phasors
    ideal_sampl_phasor = ideal_sample_phasor(f, lifetime)
    real = td_to_fd(f, timebins, counts)

    """ calculate angle offset """
    angle = ideal_sampl_phasor.angle - real.angle

    """ ratio of magnitudes -> ideal/actual """
    ratio = ideal_sampl_phasor.magnitude / real.magnitude

    return Calibration(angle=angle, scaling_factor=ratio)
