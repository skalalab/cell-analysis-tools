# MATLAB
# function [M, Phi, G, S] = phasor_calculator(f, time, decays, IRF)
# w = 2*pi*f;
# G_decay = (decays*cos(w*time))./sum(decays,2);
# S_decay = (decays*sin(w*time))./sum(decays,2);
# G_IRF = sum(IRF.*cos(w*time))/sum(IRF(:));
# S_IRF = sum(IRF.*sin(w*time))/sum(IRF(:));
# GS = [G_IRF, S_IRF; -S_IRF, G_IRF]*[G_decay'; S_decay']/(G_IRF^2 + S_IRF^2);
# G = transpose(GS(1,:));
# S = transpose(GS(2,:));
# Phi = pi-atan2(S,-G);
# M = sqrt(G.^2 + S.^2);
# end


import matplotlib.pylab as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300
from cell_analysis_tools.io import load_sdt_file

import numpy as np
from cell_analysis_tools.flim import draw_universal_semicircle


def phasor_calculator(f, time, decays, IRF):
    """
    Given an array of decay(s) the rectangular g, s and phasors angle and magnitude 
    will be computed and returned.

    Parameters
    ----------
    f : float
        Laser repetition rate.
    time : np.ndarray
        array of timebins.
    decays : np.ndarray
        Single decay or array of decays to compute phasor points for.
        Single decay should have the shape (x,),(x,t) and shape (x,y,t) for an array
    IRF : np.ndarray
        1D array capturing irf decay.

    Returns
    -------
    m : float
        magnidue of phasor.
    phi : float
        angle of phasor.
    g : float
        g coordinate (x-axis).
    s : float
        s coordinate (y-axis).


    Note
    ----
        * You cannot compare two images directly due to them having different decay shift values between images, affeting g,s,m and phi locations 
        * For proper lifetime values, background subtraction is needed by taking ~ the last 1 or 1/2 ns timebins of decay)
    
        
    .. image:: ./resources/flim_phasor_calculation.png
        :width: 600
        :alt: Image of a lifetime image as a phasor plot
        
        
    """
    
    w = 2*np.pi*f
    
    # row=pixels, cols=photons
    decays = np.reshape(decays,(-1,256))
    
    
    dim_time = len(decays.shape) -1
    G_decay = np.dot(decays, np.cos(w*time))/np.sum(decays, axis=dim_time)
    S_decay = np.dot(decays, np.sin(w*time))/np.sum(decays, axis=dim_time)
    G_IRF = np.dot(IRF,np.cos(w*time))/np.sum(IRF)
    S_IRF = np.dot(IRF, np.sin(w*time))/np.sum(IRF)
    GS = np.dot(np.array([[G_IRF, S_IRF], [-S_IRF, G_IRF]]), np.array([G_decay, S_decay]))/(G_IRF**2 + S_IRF**2)
    
    g = GS[0,:]
    s = GS[1,:]
    
    phi = np.pi-np.arctan2(s,-g)
    m = np.sqrt(g**2 + s**2)
    
    return m, phi, g, s


# for image pixels by time

if __name__ == "__main__":
    # # load irf
    irf_decay = np.loadtxt("irf.csv")   
    timebins = irf_decay[:,0]
    irf = irf_decay[:,1]
    plt.plot(timebins,irf)
    plt.title(f"IRF")
    plt.show()
    
    sdt = load_sdt_file("./bigger_beads_2.1ns.sdt")
    im_nadh = sdt.squeeze()
    decay = im_nadh.sum(axis=(0,1))
    plt.title("Beads Decay")
    plt.plot(decay)
    plt.show()
    
    plt.title("beads |  2.2ns lifetime")
    plt.imshow(im_nadh.sum(axis=2))
    plt.axis("off")
    plt.show()
    
    frequency = 0.08
    draw_universal_semicircle(laser_angular_frequency=frequency*10**9,
                              title=f"bead image as 1 decay",
                              suptitle="Phasor Plot")
    m, phi, g, s = phasor_calculator(f=frequency, time=timebins, decays=decay, IRF=irf)
    plt.scatter(g,s, label="single point", s=3)
    plt.show()
    
    
    # threshold image
    mask = im_nadh.sum(axis=2) > 1000
    im_phasor = im_nadh * mask[:,:,np.newaxis]
    
    
    # compute g,s and plot
    m, phi, g, s = phasor_calculator(f=frequency, time=timebins, decays=im_phasor, IRF=irf)
    
    draw_universal_semicircle(laser_angular_frequency=frequency*10**9,
                              suptitle="Phasor Plot",
                              title="bead image")
    plt.scatter(g,s, s=1)
    plt.show()
    
