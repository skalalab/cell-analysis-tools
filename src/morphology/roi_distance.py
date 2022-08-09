import numpy as np
from scipy import ndimage


# https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html#scipy.ndimage.distance_transform_edt
def radius_max(roi, return_index=False):
    """

    The maximum distance of any pixel in the segmented region to the closest background pixel.


    Parameters
    ----------
    roi : ndarray
        binary roi mask.
    return_index : bool, optional
        return index of pixel where max radius was found. The default is False.
        
    Returns
    -------
    float
        Max eucledian distance of roi to background.
        
    Note
    ----
    If there are multiple, it returns the first instance (see np.argmax)

    References
    ----------
    """
    distance, indices = ndimage.distance_transform_edt(roi, return_indices=True)
    idx_max_value = np.unravel_index(distance.argmax(), roi.shape)

    if return_index:
        return distance[idx_max_value], idx_max_value

    return distance[idx_max_value]


def radius_mean(roi):
    """
    Mean value the distances of all pixels in the segmented region to their closest background pixel.

    Parameters
    ----------
    roi : ndarray
        binary roi mask.

    Returns
    -------
    float
        mean distance of all the pixel distances to background

    """

    return np.mean(ndimage.distance_transform_edt(roi))


def radius_median(roi):
    """
    Median value the distances of all pixels in the segmented region to their closest background pixel.

    Parameters
    ----------
    roi : ndarray
        binary roi mask.

    Returns
    -------
    float
        median distance of all the pixels distances to background

    """
    return np.median(ndimage.distance_transform_edt(roi))


if __name__ == "__main__":

    import matplotlib as mpl
    import matplotlib.pylab as plt

    mpl.rcParams["figure.dpi"] = 300
    import numpy as np
    from skimage.morphology import disk

    roi = disk(20)
    plt.imshow(roi)
    plt.show

    # MAX RADIUS
    # https://stackoverflow.com/questions/9482550/argmax-of-numpy-array-returning-non-flat-indices
    distance, indices = ndimage.distance_transform_edt(roi, return_indices=True)
    idx_max_value = np.unravel_index(distance.argmax(), distance.shape)
    distance[idx_max_value]

    print(f"max radius: {radius_max(roi)}")
    # MEAN RADIUS
    print(f"mean radius: {radius_mean(roi)}")

    # MEDIAN RADIUS
    print(f"median radius: {radius_median(roi)}")
