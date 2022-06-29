from pathlib import Path

import matplotlib as mpl
import matplotlib.pylab as plt

mpl.rcParams["figure.dpi"] = 300
from collections import OrderedDict
from pprint import pprint

import numpy as np
import tifffile
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.morphology import binary_closing, dilation, disk, remove_small_objects


def four_color_to_unique(mask: np.ndarray, debug: bool = False) -> np.ndarray:
    """
    Converts an n-label image into a 

    Parameters
    ----------
    mask : np.ndarray
        labeled mask.
    debug : bool, optional
        show output re-labeled mask. The default is False.

    Returns
    -------
    output_mask : np.ndarray
        Mask relabeled to have unique roi values for each connected component

    """
    # determine unique labels and exclude bg(0)
    list_unique_values = list(np.unique(mask))
    list_unique_values.remove(0)  # remove bg

    # label counter and placeholder array
    roi_counter = 1
    inter_mask = np.zeros_like(mask)

    # iterate through 4 color labels
    for value in list_unique_values:
        pass
        temp_mask = np.array(mask == value, dtype=np.uint16)

        # TODO exclude these next lines, specific to Gina's masks!
        labels, _ = label(temp_mask)

        unique_labels = list(np.unique(labels))
        unique_labels.remove(0)
        for roi_value in unique_labels:
            inter_mask[labels == roi_value] = roi_counter
            roi_counter += 1  # increment roi_value

        # compute regionprops
        # sort by centroid
        props = regionprops(inter_mask)
        dict_value_location = OrderedDict({roi.label: roi.centroid for roi in props})

        # sort dict by tuple (rows, cols)
        dict_value_location_sorted = sorted(
            dict_value_location.items(), key=lambda item: (item[1][0], item[1][1])
        )

        # reoder roi values in increasing order
        roi_counter = 1
        output_mask = np.zeros_like(inter_mask)
        for roi_val, _ in dict_value_location_sorted:
            pass
            output_mask[inter_mask == roi_val] = roi_counter
            roi_counter += 1

    if debug:
        plt.imshow(output_mask)
        plt.show()

    return output_mask


if __name__ == "__main__":

    from four_color_theorem.four_colors import four_color_theorem

    mask = tifffile.imread("mask.tiff")

    # plt.imshow(mask)
    mask_fc, _ = four_color_theorem(mask)
    mask_unique = four_color_to_unique(mask_fc)

    # temp_mask = np.array(binary_closing(temp_mask, disk(1)), dtype=np.uint16)
    labeled_mask = remove_small_objects(mask_unique, min_size=30)
    plt.title(f"unique values \n unique rois: {len(np.unique(labeled_mask))-1}")
    plt.imshow(labeled_mask)
    plt.show()
