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

    ImagePhasorPoints = coll.namedtuple("ImagePhasorPoints", "points_g points_s")
    # print(image.shape)
    width, height, num_timebins = image.shape
    laser_period = 1 / f
    timebins = np.linspace(0, laser_period, num_timebins, endpoint=False)
    w = 2 * np.pi * f

    """ create 3d matrix of """
    # https://stackoverflow.com/questions/24148322/python-3d-array-times-1d-vector
    #    ones_array = np.ones((num_timebins))
    pre_comp_cos = np.cos(w * timebins)
    pre_comp_sin = np.sin(w * timebins)
    # newaxis == None, helps index/align the arrays for multiplication
    # cos_array = ones_array * pre_comp_cos[:,np.newaxis,np.newaxis]
    # sin_array = ones_array * pre_comp_sin[:,None,None]

    # image = np.ones((4,4,4)) * np.linspace(1,4,4)[:,None,None]
    # te = np.array([4,3,2,1])
    # temp = image * te[:,np.newaxis,np.newaxis]

    """image arrangement """
    axis_x, axis_y, axis_timebins = (0, 1, 2)

    #     point_g = np.sum(counts * np.cos(w * timebins)) / np.sum(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        # numerator = np.sum(image * pre_comp_cos[np.newaxis, np.newaxis,:], axis = axis_timebins)
        numerator = image
        denominator = np.sum(image, axis=axis_timebins)

        temp = np.true_divide(numerator, denominator[:, :, np.newaxis])
        temp = np.nan_to_num(temp)  # check 0/0, NaN=0
        points_g = np.sum(
            temp * pre_comp_cos[np.newaxis, np.newaxis, :], axis=axis_timebins
        )

        # points_g = np.true_divide(numerator, denominator)
        # print('division', points_g)
        # sometimes will divide by 0/0, 1/0, -1/0 check all cases
        # points_g = np.nan_to_num(points_g) # check 0/0, NaN=0
        # points_g[~ np.isfinite(points_g)] = 0 #inf and -inf = 0

    with np.errstate(divide="ignore", invalid="ignore"):
        # numerator = np.sum(image * pre_comp_sin[np.newaxis, np.newaxis, :], axis=axis_timebins)

        # temp = np.true_divide(numerator, denominator[:,:, np.newaxis])
        # temp = np.nan_to_num(temp)  # check 0/0, NaN=0
        points_s = np.sum(
            temp * pre_comp_sin[np.newaxis, np.newaxis, :], axis=axis_timebins
        )

        # denominator = np.sum(image, axis=axis_timebins)
        # points_s = np.true_divide(numerator, denominator)
        # sometimes will divide by 0/0, 1/0, -1/0 check all cases
        # points_s = np.nan_to_num(points_s)
        # points_s[~ np.isfinite(points_s)] = 0

    return ImagePhasorPoints(points_g=points_g, points_s=points_s)
