import numpy as np
from numpy.ma import masked_array


def intensity_sum(roi, im_itensity):
    """The sum of the pixel intensities within the segmented region.
    """

    roi_inverted = np.invert(np.array(roi).astype(bool))
    roi_masked = masked_array(im_itensity, mask=roi_inverted)

    return np.sum(roi_masked, axis=(0, 1))
