# Dependencies
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pylab
import tifffile
from scipy.signal import convolve

from cell_analysis_tools.image_processing import normalize
from cell_analysis_tools.io import read_asc

def bin_image(image, bin_factor):
    """
    This function takes in an lifetime image and bins 
    the decays of its histogram
    
    Parameters
    ----------

    image : ndarray
        image to bin, must be a 3d array of shape(x,y,t)
    bin_size : int
        pixel radius(including diagonals) of bins to create
    
    Returns
    -------
        binned_image : ndarray
            new image with binned pixels
    """

    #    image = image_thresholded
    #    bin_factor = 9

    # GET IMAGE DIMENSIONS
    x, y, num_timebins = image.shape
    axis_x, axis_y, axis_timebins = (0, 1, 2)
    # tif.imshow(np.sum(image,axis=axis_timebins))

    # new image dimensions
    # If not divisible by downsampling factor:
    # (y - remainder + downsampling_factor_for_padding)/downsampling factor
    # y-remainder ==> makes it divisible by downsampling factor
    # + bin_factor padds it to make it bigger, then divide
    remainder_x = x % bin_factor
    remainder_y = y % bin_factor
    subsampled_x = np.int(
        x / bin_factor
        if x % bin_factor == 0
        else (x + bin_factor - remainder_x) / bin_factor
    )
    subsampled_y = np.int(
        y / bin_factor
        if y % bin_factor == 0
        else (y + bin_factor - remainder_y) / bin_factor
    )

    """ need to pad with zeros if not divisible by bin_factor"""
    padded_x = np.int(subsampled_x * bin_factor)
    padded_y = np.int(subsampled_y * bin_factor)

    # Original image padded with zeros
    image_padded = image.copy()
    # tif.imshow(np.sum(image_padded,axis=0))

    # pad image dimensions
    col_to_add = padded_x - x
    pad_right = int(np.floor(col_to_add / 2) + padded_x % x)  # pad half pixels added
    pad_left = int(np.floor(col_to_add / 2))

    row_to_add = padded_y - y
    pad_top = int(np.floor(row_to_add / 2) + padded_y % y)
    pad_bottom = int(np.floor(row_to_add / 2))

    # np.pad((time),(rows/yaxis),(columns/xaxis))
    image_padded = np.pad(
        image_padded,
        ((0, 0), (pad_left, pad_right), (pad_top, pad_bottom)),
        "constant",
        constant_values=0,
    )

    sub_matrices = np.zeros(
        (subsampled_x, subsampled_y, num_timebins, bin_factor ** 2), dtype=object
    )

    # variable to store binned image
    binned_image = np.zeros((subsampled_x, subsampled_y, num_timebins))

    index = 0
    # GENERATE bin_factor^2 NUMBER OF SUBMATRICES
    for column in range(bin_factor):
        for row in range(bin_factor):
            temp_matrix = image_padded[
                column:padded_x:bin_factor, row:padded_y:bin_factor, :
            ]
            #            tif.imshow(np.sum(temp_matrix, axis=0))

            # add matrix to array
            sub_matrices[:, :, :, index] = temp_matrix
            index += 1

            # display sub_matrices
            # temp = sub_matrices[:,:,:,index] # last dimension is sub matrix
            # temp2 = temp.astype(float) # cast from object to float
            # plt.imshow(np.sum(temp2, axis=0)) # show submatrix

            # keep adding to the image
            binned_image = binned_image + temp_matrix

        """ improvements
        * preserve resolution like SPCImage, using kernel size
        * fft of original image
        * fft of kernel, padded to equal image size
        * take multiplication
        * convert back to time domain
        
        can we assume image is a square?
        """
        # tif.imshow(np.sum(binned_image, axis=0))

    return binned_image
