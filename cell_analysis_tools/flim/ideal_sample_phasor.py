import collections as coll

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pylab
import tifffile
from scipy.signal import convolve

from cell_analysis_tools.image_processing import normalize
from cell_analysis_tools.io import read_asc


def ideal_sample_phasor(f, lifetime):
    """
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
            
    .. code-block:: python
        
        >>print("Input is 80Mhz phasor at @ 2ns lifetime")
        >>angle, magnitude = ideal_sample_phasor(f=80e6, lifetime=2e-9)
        >>print(f"{angle=:.3f} rad | {magnitude=:.3f}")
        angle=0.788 rad | magnitude=0.705
        
    
    
    """

    Phasor = coll.namedtuple("Phasor", "angle magnitude")

    # lifetime = lifetime * 1e-12  # lifetime in ns
    w = 2 * np.pi * f
    ### simulated point values
    ideal_g = 1 / (1 + (w ** 2 * lifetime ** 2))
    ideal_s = (w * lifetime) / (1 + (w * lifetime) ** 2)

    # angle = np.arctan(ideal_s/ideal_g) # radians
    angle = np.pi - np.arctan2(ideal_s, -ideal_g)  # in radians
    magnitude = np.sqrt(ideal_g ** 2 + ideal_s ** 2)

    print("Input lifetime: ", (1 / w * ideal_s / ideal_g))
    return Phasor(angle=angle, magnitude=magnitude)

if __name__ == "__main__":
    
    from cell_analysis_tools.flim import ideal_sample_phasor
    
    angle, magnitude = ideal_sample_phasor(80e6, 2e-9)
    
    print(f"{angle=:.3f} rad | {magnitude=:.3f}")
    
    from cell_analysis_tools.flim import draw_universal_semicircle
    
    fig = draw_universal_semicircle(laser_angular_frequency=80e6)
    
    plt.scatter(0.4, 0.2, c='k')
    plt.show()
    
    
