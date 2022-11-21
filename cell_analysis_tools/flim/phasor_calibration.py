import collections as coll

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pylab
import tifffile
from scipy.signal import convolve

from cell_analysis_tools.flim import ideal_sample_phasor, lifetime_to_phasor
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
    
    
    calibration = coll.namedtuple("calibration", "angle scaling_factor")

    # calculate idea and real phasors
    ideal_sampl_phasor = ideal_sample_phasor(f, lifetime)
    real = lifetime_to_phasor(f, timebins, counts)

    """ calculate angle offset """
    angle = ideal_sampl_phasor.angle - real.angle

    """ ratio of magnitudes -> ideal/actual """
    ratio = ideal_sampl_phasor.magnitude / real.magnitude

    return calibration(angle=angle, scaling_factor=ratio)


if __name__ == "__main__":
    
    from cell_analysis_tools.io import load_sdt_file
    from cell_analysis_tools.flim import (draw_universal_semicircle,
                                          phasor_to_rectangular,
                                          rectangular_to_phasor
                                          )
    
    
    # # load irf
    irf_decay = np.loadtxt("irf.csv")   
    timebins = irf_decay[:,0]
    irf = irf_decay[:,1]
    plt.plot(timebins,irf)
        
    # load and plot original irf
    irf_offset = np.roll(irf,100)
    f = 0.08 # Ghz
    phasor = lifetime_to_phasor(f=f, timebins=timebins, counts=irf)
    phasor = lifetime_to_phasor(f=f, timebins=timebins, counts=irf_offset)
    plt.plot(timebins, irf_offset)
    
    g,s = phasor_to_rectangular(phasor.angle, phasor.magnitude)
    draw_universal_semicircle(laser_angular_frequency=80e6)
    plt.scatter(g,s, label="original point")

    
    # calibate and plot original irf
    calibration = phasor_calibration(f=80e6,
                                    lifetime=0,
                                    timebins=timebins,
                                    counts=irf)    
    
    phasor_calibrated = phasor.angle + calibration.angle, phasor.magnitude * calibration.scaling_factor
    new_g, new_s = phasor_to_rectangular(phasor_calibrated[0], phasor_calibrated[1])
    
    # PLOT 
    draw_universal_semicircle(laser_angular_frequency=80e6,
                              suptitle=f"Transformation ofsset: angle = {calibration.angle:.2f} | scaling_factor = {calibration.scaling_factor:.2f}")
    plt.scatter(g,s, label="original point")
    plt.scatter(new_g, new_s, label="calibrated point")
    plt.legend()
    
    # plot bead data
    # load image 
    sdt = load_sdt_file("./bigger_beads_2.1ns.sdt")
    im_nadh = sdt.squeeze()
    decay = im_nadh.sum(axis=(0,1))
    
    phasor = lifetime_to_phasor(f=80e6, timebins=timebins, counts=decay)
    g,s = phasor_to_rectangular(phasor.angle, phasor.magnitude)
    plt.scatter(g, s)
    
    # calibrate 
    calibration = phasor_calibration(f=80e6,
                                            lifetime=2.2e-9,
                                            timebins=timebins,
                                            counts=decay)    
    
    phasor_calibrated = phasor.angle + calibration.angle, phasor.magnitude * calibration.scaling_factor
    new_g, new_s = phasor_to_rectangular(phasor_calibrated[0], phasor_calibrated[1])
    plt.scatter(new_g, new_s)
    
    phasor2 = rectangular_to_phasor(g, s)

    #### plot image
    plt.show()
    