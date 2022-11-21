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


def rectangular_to_phasor(g, s):
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
    phasor = coll.namedtuple("phasor", "angles magnitudes")

    # calculate angles and magnitudes for all points
    angle = np.pi - np.arctan2(s, -g)
    magnitude = np.sqrt(g ** 2 + s ** 2)
    return phasor(angles=angle, magnitudes=magnitude)


if __name__ == "__main__":
    
    from cell_analysis_tools.io import load_sdt_file

    # load image 
    sdt = load_sdt_file("./resources/test_image.sdt")
    im_nadh = sdt[1,...]
    plt.imshow(im_nadh.sum(axis=2))
    plt.show()
    
    