import scipy.ndimage as ndi


def fill_and_label_rois(curr_nuclei):
    """
    Fills and labels ROI outlines using unique ints for each region.

    Parameters
    ---------- 
    param curr_nucelei : ndarray
        Current image to process

    Returns
    -------
    output : ndarray 
        ROIs filled and labeled with unique int representations

    """

    return ndi.label(ndi.binary_fill_holes(curr_nuclei))[0]
