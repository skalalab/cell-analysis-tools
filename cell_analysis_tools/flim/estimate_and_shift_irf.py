# Dependencies
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pylab
import tifffile
from scipy.signal import convolve

from cell_analysis_tools.image_processing import normalize
from cell_analysis_tools.io import read_asc


# SHIFT IRF
def estimate_and_shift_irf(decay, irf_decay, debug=False):
    """
    Estimates shift  given a decay and an IRF. 

    Parameters
    -----------
        decay : ndarray
            1D array containing decay curve
        irf_decay : ndarray
            1D array containing IRF  

        debug : bool
            Optional flag to display intermediate calculations

    Returns
    -------
        irf_decay_shifted : ndarray
            Shifted IRF as a 1d array
        shift : int
            value IRF was shifted by

    Note
    ----
        IRF and decay should not have low SNR or gradient function will produce incorrect alignment   
    
    """

    # TODO: smooth signals
    # TODO: center IRF on middle of rising decay gradient or a little before
    # peak should be near the middle of decay rise or closer to the left side
    # account for offset of decay vertically
    num_timebins = len(decay)
    # SDT DECAY
    decay_grad = np.gradient(decay)
    decay_rising = decay_grad.copy()
    decay_rising[decay_grad < 0] = 0  # take all positive gradient
    if debug:
        plt.plot(decay)
        plt.plot(decay_grad)
        plt.plot(decay_rising)
        plt.legend(["decay", "gradient", "positive gradient"])
        plt.show()

    # IRF --> Extract peak of rising
    irf_decay_grad = np.gradient(irf_decay)

    irf_rising = irf_decay_grad.copy()
    irf_rising[irf_rising < 0] = 0
    if debug:
        plt.plot(irf_decay)
        plt.plot(irf_decay_grad)
        plt.plot(irf_rising)
        plt.legend(["decay", "gradient", "positive gradient"])
        plt.show()

    # compute shift
    correlated = np.correlate(decay_rising, irf_rising, mode="full")
    peak_correlated = np.argmax(correlated)
    shift = peak_correlated - len(
        decay_rising
    )  # because length is (len_d1 + len_d2 - 1)

    irf_decay_shifted = np.roll(irf_decay, shift)

    # if debug: #visualize shifted correlation
    #     plt.plot(normalize(decay_rising))
    #     plt.plot(normalize(irf_rising))
    #     plt.plot(normalize(correlated))
    #     plt.plot(np.roll(normalize(correlated), -np.argmax(correlated) ))
    #     plt.legend(["decay rising gradient","irf rising gradient","correlated signal"])
    #     plt.show()

    if debug:  # visualize
        plt.plot(normalize(decay))
        plt.plot(normalize(irf_decay))
        plt.plot(normalize(irf_decay_shifted))
        plt.legend(["decay", "irf", "shifted irf"])
        plt.show()

    return irf_decay_shifted, shift
