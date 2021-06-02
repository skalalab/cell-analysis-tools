# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 10:21:47 2021

@author: Nabiki
"""
import tifffile
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import label
import matplotlib.pylab as plt
from pathlib import Path
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 300

def rgb2labels(im, debug=False):
    """
    Parameters
    ----------
    im : np.array
        input RGB image of rois.
    debug : bool, optional
        Enable display of intermediates. The default is False.

    Returns
    -------
    mask : TYPE
        labels mask from RGB image.

    """
    if debug:   
        plt.title("original_image")
        plt.imshow(im)
        plt.show()
    
    ##### convert RGB to intensity with unique values
    im[...,1] = im[...,1] * 2
    im[...,2] = im[...,2] * 3
    flat = np.sum(im, axis=2)
    
    if debug:
        plt.title("flattened image")
        plt.imshow(flat)
        plt.show()
    
    unique_values = np.unique(flat)
    unique_values = unique_values[1:] # remove bg
    if debug:
        print(f"unique values without bg: {unique_values}")
    
    # output mask
    mask = np.zeros(im.shape[:2])
    
    # incrementing index
    idx = 1
    for value in unique_values: # iterate through values
        im_rois = (flat==value).astype(int)
        labeled_mask, n_rois = label(im_rois)
        # iterate through labels
        for roi_label in np.unique(labeled_mask)[1:]: # exclude gb mask
            roi = (labeled_mask == roi_label).astype(int) * idx
            mask = mask + roi
            idx +=1
    if debug:
        plt.title("intermediate mask")        
        plt.imshow(mask)
        plt.show()
    
    ###### reorder index to increasing values
    
    iter_mask = mask.copy().astype(int)
    n_rows, n_cols = iter_mask.shape
    
    label_id = 1 # starting label
    for row in np.arange(n_rows):
        print(f"processing row: {row}")
        for col in np.arange(n_cols):
            if iter_mask[row,col] != 0: # mask found
                #print(f"value: {mask[row,col]}")
                roi_value = mask[row,col]
                mask[iter_mask==roi_value] = label_id # replace mask value
                iter_mask[iter_mask==roi_value] = 0 # clear out mask in iter_mask
                label_id += 1 # increment index
    if debug:
        plt.title("incrementing order of labels")
        plt.imshow(mask)
        plt.show()
        
    return mask
    
if __name__ == "__main__":
    
    # https://stackoverflow.com/questions/24780697/numpy-unique-list-of-colors-in-the-image
    # pixels_1d_array = im.reshape((-1,im.shape[2]))
    # unique_colors = np.unique(pixels_1d_array, axis=0)
    
    image_dir = Path("rgb2gray/sample_data")
    
    # path_im = image_dir / "N_photons_day22_feat2_Object Predictions_cyto_overlay.tiff"
    path_im = image_dir / "N_photons_day22_feat2_Object Predictions_cyto.tiff"
    
    debug = True
    im = tifffile.imread(path_im)
    mask = rgb2labels(im, debug=True)
    
    unique_colors = np.unique(pixels_1d_array, axis=0)
