from pathlib import Path
from skimage.morphology import dilation, disk
import re

import matplotlib.pylab as plt 
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import tifffile

    
# using skimage
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter

import numpy as np

import cell_analysis_tools as cat

#%%
def mask_to_outlines(mask, im = None, binary_mask=False, debug=False):
    """
    Creates an outline of regions based on the input labels mask.

    Parameters
    ----------
    mask : np.ndarray
        Labels mask to convert to outlines.
    im : np.ndarray, optional
        DESCRIPTION. The default is None.
    binary_mask : bool, optional
        Determine if returned array should be boolean. The default is False.
    debug : TYPE, optional
        Displays intermediate images/masks for debugging. The default is False.

    Returns
    -------
    mask_outline : np.ndarray
        Array containing outlines of input labeled mask.
        
    
    .. image:: ./resources/visualization-mask_to_outlines.png
        :width: 600
        :alt: Image showing original mask and outlines after being run through this function

    """
    if debug:
        fig, ax = plt.subplots()
        ax.imshow(im, cmap=plt.cm.gray)
    
    mask_outline = np.zeros_like(mask)
    
    
    for idx, label in enumerate(np.unique(mask)[1:]): # skip bg mask
        pass
        one_roi = (mask == label)
        contours = find_contours(one_roi)
    
        # plot over figure
        for contour in contours:
            if debug:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=1)
        
            rr, cc = polygon_perimeter(contour[:, 1], contour[:, 0], mask_outline.shape)
            
            if binary_mask:
                mask_outline[cc,rr] = 1
            else:
                mask_outline[cc,rr] = idx
    plt.show()
    if debug:
        cat.visualization.compare_images("mask", mask, 'outlines', mask_outline)

    if binary_mask:
        mask_outline = mask_outline.astype(bool)
    return mask_outline
    
#%%

if __name__ == "__main__":
    pass
    
    from skimage.data import binary_blobs
    from skimage.morphology import label
    
    mask = binary_blobs(length=256, volume_fraction=0.2)
    
    mask_labeled = label(mask)
    plt.imshow(mask_labeled)
    plt.show()
    
    
    
    mask_outlines = mask_to_outlines(mask_labeled) # , binary_mask=True
    plt.imshow(mask_outlines, vmax=mask_outlines.max())
    plt.show()

    

    