 # -*- coding: utf-8 -*-
"""
Created on Fri Mar  11 8:32::01 2019
@author: Emmanuel Contreras Guzman
"""

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



def lifetime_image_to_rectangular_points(f, image):
    """
    This function takes all the pixels/histograms
    of an image and plots them in the phasor plot

    Parameters
    ----------
    f : frequency 
        laser angular frequency
    image : ndarray
        3d array representation of image, where the axis are [t,x,y]

    Returns
    -------

    ImagePhasorPoints : ndarray 
        object holding array of points
            points_g - 2d array of g coordinates
            points_s - 2d array of s coordinates
    """

    # use the code below created a 3d "image" of histograms with (np.cos(w * timebins)) pre computed
    # then do matrix multiplication to calculate the g and s for every pixel and then plot
    
    ImagePhasorPoints = coll.namedtuple('ImagePhasorPoints', 'points_g points_s')
    # print(image.shape)
    width, height, num_timebins = image.shape
    laser_period = 1/f
    timebins = np.linspace(0, laser_period, num_timebins, endpoint=False)
    w = 2 * np.pi * f
    
    ''' create 3d matrix of ''' 
    #https://stackoverflow.com/questions/24148322/python-3d-array-times-1d-vector
#    ones_array = np.ones((num_timebins))
    pre_comp_cos = np.cos(w * timebins)
    pre_comp_sin = np.sin(w * timebins)
    #newaxis == None, helps index/align the arrays for multiplication
    #cos_array = ones_array * pre_comp_cos[:,np.newaxis,np.newaxis]
    #sin_array = ones_array * pre_comp_sin[:,None,None]
    
    #image = np.ones((4,4,4)) * np.linspace(1,4,4)[:,None,None]
    #te = np.array([4,3,2,1])
    #temp = image * te[:,np.newaxis,np.newaxis]

    '''image arrangement '''
    axis_x, axis_y, axis_timebins = (0, 1, 2)

#     point_g = np.sum(counts * np.cos(w * timebins)) / np.sum(counts)
    with np.errstate(divide='ignore', invalid='ignore'):
        # numerator = np.sum(image * pre_comp_cos[np.newaxis, np.newaxis,:], axis = axis_timebins)
        numerator = image
        denominator = np.sum(image, axis=axis_timebins)

        temp = np.true_divide(numerator, denominator[:,:, np.newaxis])
        temp = np.nan_to_num(temp) # check 0/0, NaN=0
        points_g = np.sum(temp * pre_comp_cos[np.newaxis, np.newaxis, :], axis=axis_timebins)

        # points_g = np.true_divide(numerator, denominator)
        # print('division', points_g)
        # sometimes will divide by 0/0, 1/0, -1/0 check all cases 
        # points_g = np.nan_to_num(points_g) # check 0/0, NaN=0
        # points_g[~ np.isfinite(points_g)] = 0 #inf and -inf = 0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # numerator = np.sum(image * pre_comp_sin[np.newaxis, np.newaxis, :], axis=axis_timebins)

        # temp = np.true_divide(numerator, denominator[:,:, np.newaxis])
        # temp = np.nan_to_num(temp)  # check 0/0, NaN=0
        points_s = np.sum(temp * pre_comp_sin[np.newaxis, np.newaxis, :], axis=axis_timebins)

        # denominator = np.sum(image, axis=axis_timebins)
        # points_s = np.true_divide(numerator, denominator)
        # sometimes will divide by 0/0, 1/0, -1/0 check all cases 
        # points_s = np.nan_to_num(points_s)
        # points_s[~ np.isfinite(points_s)] = 0

    return ImagePhasorPoints(points_g=points_g, points_s=points_s)

def td_to_fd(f, timebins, counts):
    ''' Time to frequency domain transformation

    Parameters
    ---------- 
        f  : int 
            laser repetition angular frequency
        timebins : ndarray
            numpy array of timebins
        counts : ndarray
            photon counts of the histogram()

    Returns
    -------
        angle  : float 
            angle in radians
        magnitude  : float 
            magnitude of phasor
    '''
    w = 2 * np.pi * f  # f
    Phasor = coll.namedtuple('Phasor', 'angle magnitude') #nameddtuple 
    # pylab.plot(timebins,counts)
    
    ## convert to phasor rectangular
    point_g = np.sum(counts * np.cos(w * timebins))/np.sum(counts)
    point_s = np.sum(counts * np.sin(w * timebins))/np.sum(counts)
    
    #https://software.intel.com/en-us/forums/archived-visual-fortran-read-only/topic/313067
    # 0.5*TWOPI-ATAN2(Y,-X)
    #angle = AMOD(ATAN2(y,x)+TWOPI,TWOPI)
    angle = np.pi-np.arctan2(point_s, -point_g)
    magnitude = np.sqrt(point_g**2 + point_s**2)

    return Phasor(angle=angle, magnitude=magnitude)


def phasor_calibration(f, lifetime, timebins, counts):
    ''' Phasor Plot Calibration

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
    '''
    Calibration = coll.namedtuple('Calibration', 'angle scaling_factor')

    # calculate idea and real phasors
    ideal_sampl_phasor = ideal_sample_phasor(f, lifetime)
    real = td_to_fd(f, timebins, counts)
    
    ''' calculate angle offset '''
    angle = ideal_sampl_phasor.angle - real.angle    
    
    ''' ratio of magnitudes -> ideal/actual '''
    ratio = ideal_sampl_phasor.magnitude / real.magnitude
    
    return Calibration(angle=angle,  scaling_factor=ratio)
    
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


def rectangular_to_phasor_point(g, s):
    print('not implemented yet')

    angle = np.pi-np.arctan2(s,-g)
    magnitude = np.sqrt(g**2 + s**2)
    return 


def rectangular_to_phasor_lifetimes_array(g, s):
    ''' 
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
    '''
    PhasorLifetimesArray = coll.namedtuple('PhasorLifetimesArray', 'angles magnitudes')

    # calculate angles and magnitudes for all points
    angles = np.pi-np.arctan2(s, -g)
    magnitudes = np.sqrt(g**2 + s**2)
    return PhasorLifetimesArray(angles=angles, magnitudes=magnitudes)


def phasor_to_rectangular_lifetimes_array(angles, magnitudes):
    ''' Takes in two arrays of angles and magnitudes
    and converts them to array of g and s points
    
    Parameters
    ----------
        f : int 
            laser angular frequency
        angles : float 
            array of angles
        mangitudes : float
            array of magnitudes
    Returns
    -------
        RectangularLifetimesArray : Phasor collection object
            object with g and s array of lifetimes
                g - array of g-coordinates(x-coordinates)
                s - array of s-coordinates (y-coordinates)
    '''
    RectangularLifetimesArray = coll.namedtuple('RectangularLifetimesArray', 'g s')

    #calculate g and s
    g = magnitudes * np.cos(angles)
    s = magnitudes * np.sin(angles)
    
    return RectangularLifetimesArray(g=g, s=s)
    
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


def bin_image(image, bin_factor):
    '''
    This function takes in an image and bins it's histograms
    
    
    Parameters
    ----------

    image : ndarray
        image to bin, must be a 3d array of shape(t,x,y)
    bin_size : int
        pixel radius(including diagonals) of bins to create
    
    Returns
    -------
        binned_image : ndarray
            new image with binned pixels
    '''

#    image = image_thresholded
#    bin_factor = 9

    #GET IMAGE DIMENSIONS
    x, y, num_timebins = image.shape
    axis_x, axis_y, axis_timebins = (0,1,2)
    # tif.imshow(np.sum(image,axis=axis_timebins))
    
    # new image dimensions
    # If not divisible by downsampling factor: 
    #(y - remainder + downsampling_factor_for_padding)/downsampling factor
    #y-remainder ==> makes it divisible by downsampling factor
    # + bin_factor padds it to make it bigger, then divide
    remainder_x = x%bin_factor
    remainder_y = y%bin_factor
    subsampled_x = np.int(x/bin_factor if x % bin_factor == 0 else(x + bin_factor - remainder_x)/bin_factor)
    subsampled_y = np.int(y/bin_factor if y % bin_factor == 0 else(y + bin_factor - remainder_y)/bin_factor)
    
    ''' need to pad with zeros if not divisible by bin_factor'''
    padded_x = np.int(subsampled_x * bin_factor)
    padded_y = np.int(subsampled_y * bin_factor)
    
    # Original image padded with zeros
    image_padded = image.copy()
    # tif.imshow(np.sum(image_padded,axis=0))
    
    #pad image dimensions
    col_to_add = padded_x-x
    pad_right = int(np.floor(col_to_add/2) + padded_x%x) # pad half pixels added
    pad_left = int(np.floor(col_to_add/2))
    
    row_to_add = padded_y-y
    pad_top = int(np.floor(row_to_add/2) + padded_y%y)
    pad_bottom = int(np.floor(row_to_add/2))
    
    #np.pad((time),(rows/yaxis),(columns/xaxis))
    image_padded = np.pad(image_padded,((0,0),(pad_left,pad_right),(pad_top,pad_bottom)),'constant',constant_values=0)
    
    sub_matrices = np.zeros((subsampled_x,subsampled_y, num_timebins,  bin_factor**2),dtype=object)
    
    #variable to store binned image
    binned_image = np.zeros((subsampled_x, subsampled_y, num_timebins))

    index = 0
    #GENERATE bin_factor^2 NUMBER OF SUBMATRICES
    for column in range(bin_factor):
        for row in range(bin_factor):
            temp_matrix = image_padded[column:padded_x:bin_factor , row:padded_y:bin_factor, :]
#            tif.imshow(np.sum(temp_matrix, axis=0))
            
            # add matrix to array
            sub_matrices[:,:,:,index] = temp_matrix
            index += 1
            
            # display sub_matrices
            #temp = sub_matrices[:,:,:,index] # last dimension is sub matrix
            #temp2 = temp.astype(float) # cast from object to float
            #plt.imshow(np.sum(temp2, axis=0)) # show submatrix
                
            #keep adding to the image
            binned_image = binned_image + temp_matrix
     
        ''' improvements
        * preserve resolution like SPCImage, using kernel size
        * fft of original image
        * fft of kernel, padded to equal image size
        * take multiplication
        * convert back to time domain
        
        can we assume image is a square?
        '''
        #tif.imshow(np.sum(binned_image, axis=0))
   
    return binned_image

    

def draw_universal_semicircle(figure, laser_angular_frequency, title='Phasor Plot',  debug=False):
    """
    Draws the universal semicircle over the give figure.

    Parameters
    ----------
        figure : matplotlib figure object
            figure object to draw semicircle over
        laser_angular_frequency : int
            rep rate or laser angular frequency, affects position of labeled points

    Returns
    -------
        None
    """
    
    plt.title(title)
    ''' get universal semicircle values for this rep rate '''
    x_circle, y_circle, g, s, lifetime_labels = universal_semicircle_series(laser_angular_frequency)

    # Labels and axis of phasor plot
    # figure = plt.figure()
    figure.suptitle(title)
    plt.xlabel('g', fontsize=20)
    plt.ylabel('s', fontsize=20)

    # add circle and lifetime estimates
    plt.plot(x_circle, y_circle, '-', color='teal')
    plt.plot(g, s, '.', color='magenta')

    if debug:
        print('type: ', type(lifetime_labels), ' labels: ', lifetime_labels)

    ''' ADD LIFETIME LABELS '''
    for i, txt in enumerate(lifetime_labels):
        # self.ax.annotate(txt, (g[i], s[i]))
        plt.annotate(txt, (g[i], s[i]))
    plt.show()


def universal_semicircle_series(frequency):
    """

    Given the frequency this function will return x and y points for plotting the
    universal semicircle, as well as the corresponding g and s values for
    common lifetimes between 0.5ns to 10ns

    Parameters
    ----------
        frequency {float} : angular laser repetition frequency
    
    Returns
    -------

        x_circle : float
            x coordinates for universal semicircle
        y_circle : float
            y coordinates for universal semicircle
        g : float
            x coordinates for labeled lifetimes
        s : float
            y coordinates for labeled lifetimes
        lifetime_labels : list
            labels to be applied to the (g,s) coordinates
    """
    x_coord = np.linspace(0, 1, num=1000)
    y_circle = np.sqrt(x_coord - x_coord ** 2)
    x_circle = x_coord ** 2 + y_circle ** 2

    omega = 2.0 * np.pi * frequency  # modulation frequency
    tau = np.asarray([0.5e-9, 1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 8e-9, 9e-9, 10e-9])  # lifetimes in ns
    g = 1 / (1 + np.square(omega) * np.square(tau))
    s = (omega * tau) / (1 + np.square(omega) * np.square(tau))
    lifetime_labels = ['0.5ns', '1ns', '2ns', '3ns', '4ns', '5ns', '6ns', '7ns', '8ns', '9ns', '10ns']
    return x_circle, y_circle, g, s, lifetime_labels




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
    #SDT DECAY
    decay_grad = np.gradient(decay)
    decay_rising = decay_grad.copy()
    decay_rising[decay_grad < 0] = 0 # take all positive gradient
    if debug:
        plt.plot(decay)
        plt.plot(decay_grad)
        plt.plot(decay_rising)
        plt.legend(["decay","gradient","positive gradient"])
        plt.show()
    
    # IRF --> Extract peak of rising
    irf_decay_grad = np.gradient(irf_decay)

    irf_rising = irf_decay_grad.copy()
    irf_rising[irf_rising < 0] = 0
    if debug:
        plt.plot(irf_decay)
        plt.plot(irf_decay_grad)
        plt.plot(irf_rising)
        plt.legend(["decay","gradient","positive gradient"])
        plt.show()

    #compute shift 
    correlated = np.correlate(decay_rising, irf_rising, mode="full")
    peak_correlated = np.argmax(correlated)
    shift = peak_correlated - len(decay_rising) # because length is (len_d1 + len_d2 - 1)

    
    irf_decay_shifted = np.roll(irf_decay, shift)

    # if debug: #visualize shifted correlation   
    #     plt.plot(normalize(decay_rising))
    #     plt.plot(normalize(irf_rising))
    #     plt.plot(normalize(correlated))
    #     plt.plot(np.roll(normalize(correlated), -np.argmax(correlated) ))
    #     plt.legend(["decay rising gradient","irf rising gradient","correlated signal"])
    #     plt.show()
        
    if debug: # visualize 
        plt.plot(normalize(decay))
        plt.plot(normalize(irf_decay))
        plt.plot(normalize(irf_decay_shifted))
        plt.legend(["decay","irf","shifted irf"])
        plt.show()   
    
    
    return irf_decay_shifted, shift

if __name__ == "__main__":
    
    from pathlib import Path
    from flim_tools.io import load_sdt_file
    import matplotlib.pylab as plt
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 600
    import numpy as np
    from flim_tools.image_processing import bin_sdt

    
    # variables
    laser_angular_frequency = 80e6
    
    ### T Cells 
    # path_sdt = Path("C:/Users/Nabiki/Desktop/data/T_cells-paper/Data/011118 - Donor 4/SDT Files/Tcells-001.sdt")
    # sdt_im = load_sdt_file(path_sdt).squeeze()
    # n_rows, n_cols, n_timebins = sdt_im.shape
    # integration_time = 1 / laser_angular_frequency
    # timebins = np.linspace(0, integration_time, n_timebins, endpoint=False)
    # decay = np.sum(sdt_im, axis=(0,1))
    # plt.plot(decay)
    # plt.show()
    # plt.imshow(sdt_im.sum(axis=2))
    # plt.show()
    
    
    HERE = Path(__file__).resolve().parent
    ########### neutrohpils 
    # working_dir = Path(r"C:\Users\Nabiki\Desktop\development\flim_tools\flim_tools\example_data\t_cell".replace('\\','/'))
    # path_sdt = working_dir / "Tcells-001.sdt"
    # path_sdt = Path(HERE.parent / "example_data/neutrophils/Neutrophils-021_NADH.sdt")
    
    
    ###### LOAD SDT FILE 
    #Kelsey IRF's
    # irf = tifffile.imread( Path(HERE.parent / "example_data/neutrophils/Neutrophils-021_IRF.tiff"))
    # irf = read_asc("Z:/0-Projects and Experiments/KM - OMI Phasor Plots/40x_WID_2019Mar_IRF.asc")
    irf = read_asc(HERE / "example_data/irf_40xW_02_dec2017_IRF.asc")
    
    irf_timebins = irf[:,0] * 1e-9 # timebins in ns
    irf_decay = irf[:,1] # photons count
    


    ###### LOAD SDT FILE 
    path_sdt = Path(HERE / "example_data/EPC16_Day8_4n-063/LifetimeData_Cycle00001_000001.sdt")

    im_sdt = load_sdt_file(path_sdt).squeeze()
    n_rows, n_cols, n_timebins = im_sdt.shape
    integration_time = 1 / laser_angular_frequency
    timebins = np.linspace(0, integration_time, n_timebins, endpoint=False)
    decay = np.sum(im_sdt, axis=(0,1))
    plt.plot(decay)
    plt.show()
    plt.imshow(im_sdt.sum(axis=2))
    plt.show()
    
    

        
    # offset decay

    
    # REMOVE DECAY OFFSET
    

    # 7x7 bin
    im_sdt_binned = bin_sdt(im_sdt, bin_size=3, debug=True)    
    
    #threshold decays 
    decays = im_sdt_binned[im_sdt_binned.sum(axis=2)>2000]
    
    
    # calculate shift here after removing bg
    
    # show First 100 decays
    # for d in decays[:100]:
    #     plt.plot(d)
    # plt.show()
    
    
    # compute calibration after irf aligned
    irf_lifetime = 0
    calibration = phasor_calibration(laser_angular_frequency, 
                                     irf_lifetime, 
                                     irf_timebins, 
                                     irf_decay)
   


    
    # COMPUTE G AND S VALUES 
    array_phasor = [td_to_fd(laser_angular_frequency, irf_timebins, decay) for decay in decays]

   
    # compute g and s for 
    list_gs = [phasor_to_rectangular_point(ph.angle + calibration.angle ,
                                            ph.magnitude * calibration.scaling_factor) 
                for ph in array_phasor]
    
    g = [point.g for point in list_gs]
    s = [point.s for point in list_gs]
    counts = decays.sum(axis=1)
    
    # plot
    figure = plt.figure()
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.axis('equal')
    plt.scatter(g, s, s=1,  cmap='viridis_r', alpha=1)  # s=size, c= colormap_data, cmap=colormap to use    
    # plt.colorbar()
    draw_universal_semicircle(figure, laser_angular_frequency)




#%%
#### Kaivalya

# Z:\0-Projects and Experiments\KM - OMI Phasor Plots\OMI images
