import numpy as np
from skimage.measure import regionprops, label
from numpy.ma import masked_array


def _return_regionprops(roi):

    im_labeled = label(roi)
    im_regionprops = regionprops(im_labeled)
    assert len(im_regionprops) == 1, "Error: multi roi mask passed"
    return im_regionprops[0]  # return single roi's props


def area(roi: np.ndarray):
    """
    Number of pixels in a segmented region.
    
    Parameters
    ----------
    roi : ndarray
        roi of region to copute area

    Returns
    -------
    area : int
        Number of pixels in the mask
    
    Examples
    --------
    >>> from skiamge.draw import ellipse
    >>> import numpy as np
    >>> idx_rows, idx_cols = ellipse(20,20, 5,7)
    >>> shape_ellipse = np.zeros((40,40))
    >>> shape_ellipse[idx_rows, idx_cols] = 1
    >>> area_ellipse = int(np.sum(shape_ellipse))
    >>> area_ellipse
    105
    """

    return _return_regionprops(roi).area


def major_axis_length(roi):
    """
    Number of pixels on the major axis of the fitted ellipse.

    Parameters
    ----------
    roi : ndarray
        ndarray with single roi.

    Returns
    -------
    int
        numbe of pixels in the major axis

    """
    return _return_regionprops(roi).major_axis_length


def minor_axis_length(roi):
    """
    Number of pixels on the minor axis of the fitted ellipse.

    Parameters
    ----------
    roi : ndarray
        ndarray with single roi.

    Returns
    -------
    int
         numbe of pixels in the minor axis

    """

    return _return_regionprops(roi).minor_axis_length


def eccentricity(roi):
    """
    Ratio of the distance between the foci of the ellipse to its major axis length.

    Parameters
    ----------
    roi : ndarray
        ndarray with single roi.

    Returns
    -------
    int
        eccentricity value

    """

    return _return_regionprops(roi).eccentricity


def orientation(roi):
    """
    Angle between the 0th axis (rows) and the major axis of the 
    ellipse that has the same second moments as the region, 
    ranging from -pi/2 to pi/2 counter-clockwise.
    Parameters
    ----------
    roi : ndarray
        ndarray with single roi.

    Returns
    -------
    float
        Angle between major axis of ellipse and the x-axis.

    """

    return _return_regionprops(roi).orientation


def solidity(roi):
    """
    Ratio of the area of the segmented region over the area of 
    the convex hull of the segmentation.

    Parameters
    ----------
    roi : ndarray
        ndarray with single roi.

    Returns
    -------
    float
        solidity value

    """

    return _return_regionprops(roi).solidity


def extent(roi):
    """
    Ratio of the area of the segmented region over the area of 
    the bounding box of the segmented region.


    Parameters
    ----------
    roi : ndarray
        ndarray with single roi.

    Returns
    -------
    float
        extent ratio value

    """

    return _return_regionprops(roi).extent


def perimeter(roi):
    """
    Number of pixels in the contour of the segmented region.

    Parameters
    ----------
    roi : ndarray
        ndarray with single roi.

    Returns
    -------
    int
        number of pixels in perimeter.

    """
    return _return_regionprops(roi).perimeter


def average_intensity(roi):
    """The average of pixel intensities within the segmented region.

    """
    return _return_regionprops(roi).intensity_mean


def integrated_intensity(roi, im_itensity):
    """The sum of the pixel intensities within the segmented region.

    """
    roi_inverted = np.array(roi).astype(bool).invert()
    roi_masked = masked_array(im_itensity, mask=roi_inverted)

    yield np.sum(roi_masked, axis=(0, 1))


def mitochondria_morphological_class(roi):
    """The punctate, swollen, and networked morphologies.
    """
    pass


if __name__ == "__main__":

    from skimage.draw import ellipse
    import matplotlib.pylab as plt
    import matplotlib as mpl

    mpl.rcParams["figure.dpi"] = 300

    idx_rows, idx_cols = ellipse(20, 20, 5, 7)
    shape_ellipse = np.zeros((40, 40))
    shape_ellipse[idx_rows, idx_cols] = 1

    plt.imshow(shape_ellipse)
    plt.show()

    area(shape_ellipse)

    im_labeled = label(shape_ellipse)
    plt.imshow(im_labeled)
    plt.show()

    im_regionprops = regionprops(im_labeled)

    _return_regionprops(shape_ellipse).area
