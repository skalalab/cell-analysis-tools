from pathlib import Path
import re
import tifffile
import matplotlib.pylab as plt
import numpy as np

from skimage.measure import regionprops
from skimage.morphology import closing, disk, remove_small_objects, label
from cell_analysis_tools.visualization import compare_images
from tqdm import tqdm
import tifffile

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
 #%%


def remove_small_areas_fill_regions(mask : np.array,
                                    region_min_size : int = 100,
                                    footprint_area_closing : int = 2,
                                    debug=False):
    """
    Function was created to clean up masks that may have stray pixels or regions. 

    Parameters
    ----------
    mask : np.array
        original mask.
    region_min_size : int, optional
        minimum size of connected component to be removed. The default is 100.
    footprint_area_closing : int, optional
        When merging images radius of disk to use. The default is 2.
    debug : TYPE, optional
        Enable/Disable display of intermediate images for debugging function. The default is False.

    Returns
    -------
    np.array of revised mask

    """    

    if debug:
        plt.title("input mask")
        plt.imshow(mask)
        plt.show()
    
    mask_no_small_objects = remove_small_objects(mask, region_min_size)

    if debug:
        plt.title("after removed small objects ")
        plt.imshow(mask_no_small_objects)
        plt.show()
     
    mask_revised = np.zeros_like(mask)
    for label_value in np.unique(mask_no_small_objects):
        pass
    
        # isolate roi
        mask_roi = mask == label_value
        
        # get largest region
        mask_roi_labels_mask = label(mask_roi)
        roi = sorted(regionprops(mask_roi_labels_mask), key=lambda r : r.area, reverse=True)[0]
        mask_largest = mask_roi_labels_mask == roi.label

        # fill holes in roi
        mask_closing = closing(mask_largest,footprint=disk(footprint_area_closing))
        mask_revised[mask_closing] = label_value
    
    if debug:
        plt.title("final output")
        plt.imshow(mask_revised)
        plt.show()

    return mask_revised

if __name__== "__main__":
    
    path_mask = Path(r"/mnt/Z/0-Projects and Experiments/TQ - cardiomyocyte maturation/datasets/H9/DAY 90/masks/H9_DAY_90_2n_photons_mask_cell.tiff")

    mask = tifffile.imread(path_mask)
    plt.imshow(mask)
    plt.show()
    
    mask_revised = remove_small_areas_fill_regions(mask,
                                                   region_min_size = 10,
                                                   footprint_area_closing=10,
                                                   debug=True)
    plt.imshow(mask_revised)
    plt.show()



