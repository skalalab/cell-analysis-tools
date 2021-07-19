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

def lifetime_image_to_rectangular_points(f, image):
    """ This function takes all the pixels/histograms
    of an image and plots them in the phasor plot
    Takes in a
    inputs:
        f - laser angular frequency
        image -  3d array representation of image, where the axis are [t,x,y]
    outputs:
        ImagePhasorPoints - object holding array of points
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
        input
            - f: laser repetition angular frequency
            - timebins: numpy array of timebins
            - counts: photon counts of the histogram()
        returns: 
            - angle (rad)
            - magnitude of phasor
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
        input 
            - f: laser repetition angular frequency
            - lifetime: lifetime of known sample in ns (single exponential decay)
            - timebins: timebins of samples
            - counts: photon counts of histogram
        returns
            - angle_offset: difference in angle between known sample and actual
            - magnitude_offset: difference in magnitude between known sample and actual
        Note: if no timebins or histograms passed then returns angle and phase of
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
    ''' get x position of phasor
    input:
        - angle
        - magnitude
    returns:
        - g: x axis coordinate
        - s: y axis coordinate
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
    inputs:
        g - array of g coordinates (x-coordinates)
        s - array of s coordinates (y-coordinates)
    outputs:
        PhasorArray object
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
    input:
        f - laser angular frequency
        angles - array of angles
        mangitudes - array of magnitudes
    output
        RectangularLifetimesArray - object with g and s array of lifetimes
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
    input:
        - f: laser rep rate
        - lifetime: lifetime of desired single exponential sample
    returns:
        angle - angle of ideal sample
        magnitude - magnitude of ideal sample
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
    input:
        image - image to bin, must be a 3d array of shape(t,x,y)
        bin_size - pixel radius(including diagonals) of bins to create
    output:
        binned_image - new image with binned pixels
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

    :param frequency: angular laser repetition frequency
    :returns:
    x_circle: x coordinates for universal semicircle
    y_circle: y coordinates for universal semicircle
    g: x coordinates for labeled lifetimes
    s: y coordinates for labeled lifetimes
    lifetime_labels: labels to be applied to the (g,s) coordinates
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


if __name__ == "__main__":
    
    from pathlib import Path
    from flim_tools.io import load_sdt_file
    import matplotlib.pylab as plt
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 600
    import numpy as np
    
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
    path_sdt = Path(HERE.parent / "example_data/neutrophils/Neutrophils-021_NADH.sdt")

    im_sdt = load_sdt_file(path_sdt).squeeze()
    n_rows, n_cols, n_timebins = im_sdt.shape
    integration_time = 1 / laser_angular_frequency
    timebins = np.linspace(0, integration_time, n_timebins, endpoint=False)
    decay = np.sum(im_sdt, axis=(0,1))
    plt.plot(decay)
    plt.show()
    plt.imshow(im_sdt.sum(axis=2))
    plt.show()

    from flim_tools.image_processing import bin_sdt

    # 7x7 bin
    im_sdt_binned = bin_sdt(im_sdt, bin_size=3, debug=True)    
    
    #threshold decays 
    decays = im_sdt_binned[im_sdt_binned.sum(axis=2)>1000]
        
    
    # show First 100 decays
    # for d in decays[:100]:
    #     plt.plot(d)
    # plt.show()
    
   
    #Kelsey IRF's
    irf = tifffile.imread( Path(HERE.parent / "example_data/neutrophils/Neutrophils-021_IRF.tiff"))
    irf_timebins = irf[:,0] * 1e-9 # timebins in ns
    irf_decay = irf[:,1] # photons count
    plt.plot(irf_timebins, irf_decay)
    plt.show()
    
    irf_lifetime = 0
    calibration = phasor_calibration(laser_angular_frequency, 
                                     irf_lifetime, 
                                     irf_timebins, 
                                     irf_decay)
    # enable to plot IRF 
    # decays = irf_decay.reshape((1,-1))
    
    # compute g and s  

    array_phasor = [td_to_fd(laser_angular_frequency, irf_timebins, decay) for decay in decays]

   
    # # compute g and s for 
    list_gs = [phasor_to_rectangular_point(ph.angle + calibration.angle ,
                                            ph.magnitude * calibration.scaling_factor) 
                for ph in array_phasor]
    
    g = [point.g for point in list_gs]
    s = [point.s for point in list_gs]
    counts = decays.sum(axis=1)
    
    # plot
    figure = plt.figure()
    plt.ylim([0,0.6])
    plt.xlim([0,1])
    plt.scatter(g, s, s=2, c=counts, cmap='viridis_r', alpha=0.7)  # s=size, c= colormap_data, cmap=colormap to use    
    plt.colorbar()
    draw_universal_semicircle(figure, laser_angular_frequency)




#%%
#### Kaivalya

# Z:\0-Projects and Experiments\KM - OMI Phasor Plots\OMI images
