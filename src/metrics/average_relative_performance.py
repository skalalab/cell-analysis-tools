import numpy as np

from .helper import _validate_array_and_make_bool


def average_relative_performance(mask_gt1, mask_gt2, mask_pred):
    """
    Returns average relative performance of a predicted mask against 
    two ground truth masks from two different users
    
    Taken from mito hacker paper `https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7642274/`

    ((Mito Catcher vs. User 1 + Mito Catcher vs. User 2)/(User 1 vs. User 2))
    
    Parameters
    ----------
    mask_gt1 : ndarray
        ground truth mask 1.
    mask_gt2 : ndarray
        ground truth mask 2.
    mask_predicted : ndarray
        predicted mask.

    Returns
    -------
    float
        average performance score.

    """
    mask_gt1 = _validate_array_and_make_bool(mask_gt1)
    mask_gt2 = _validate_array_and_make_bool(mask_gt2)
    mask_predicted = _validate_array_and_make_bool(mask_pred)

    return (
        np.sum(np.logical_and(mask_predicted, mask_gt1))
        + np.sum(np.logical_and(mask_predicted, mask_gt2))
    ) / (2 * np.sum((np.logical_and(mask_gt1, mask_gt2))))
