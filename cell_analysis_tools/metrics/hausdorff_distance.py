from scipy.spatial.distance import directed_hausdorff

from .helper import _validate_array_and_make_bool


def hausdorff_distance(mask_pred, mask_gt):
    """
    Calculates the Hausdorff distance for a given image, provided a ground truth. Both arrays must have the same number of columns. 
    see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
    Parameters
    ----------
    mask_pred : np.ndarray
        predicted segmentation mask
    mask_gt : np.ndarray
        ground truth mask

    Returns
    -------
        ddouble
            The directed Hausdorff distance between arrays u and v,
        index_1int
            index of point contributing to Hausdorff pair in u
        index_2int
            index of point contributing to Hausdorff pair in v

    """

    mask_pred = _validate_array_and_make_bool(mask_pred)
    mask_gt = _validate_array_and_make_bool(mask_gt)

    return directed_hausdorff(mask_pred, mask_gt)
