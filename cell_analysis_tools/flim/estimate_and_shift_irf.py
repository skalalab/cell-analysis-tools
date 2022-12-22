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
        IRF and decay should NOT have low SNR or gradient function will produce incorrect alignment   
    
    
    .. image:: ./resources/flim_estimate_and_shift_irf.png
        :width: 800
        :alt: Image of the estimated placement of IRF for a decay
        
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
        plt.title("Decay Gradient")
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
        plt.title("IRF Gradient")
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
        plt.title(f"Shift estimation \n{shift} ps")
        plt.xlabel("Time(ps)")
        plt.ylabel("Counts")
        plt.plot(normalize(decay))
        plt.plot(normalize(irf_decay))
        plt.plot(normalize(irf_decay_shifted))
        plt.legend(["decay", "irf", "shifted irf"])
        plt.show()

    return irf_decay_shifted, shift


if __name__ == "__main__":
    pass

    from cell_analysis_tools.io import load_sdt_file
    
    # load irf
    irf_decay = np.loadtxt("irf.csv")    
    plt.plot(irf_decay[:,0],irf_decay[:,1])
    
    # load image 
    sdt = load_sdt_file("./resources/test_image.sdt")
    im_nadh = sdt[1,...]
    # plt.imshow(im_nadh.sum(axis=2))
    # plt.show()
    
    decay = im_nadh.sum(axis=(0,1))
    decay = np.roll(decay,10)
    # plt.plot(decay)
    # plt.show()
    
    # visualize and plot IRF
    irf_decay_shifted, shift = estimate_and_shift_irf(decay, irf_decay[:,1], debug=True)
    
    
    
    
    
    