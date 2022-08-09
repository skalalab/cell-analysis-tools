import numpy as np
from .helper import _validate_array_and_make_bool


def jaccard(mask_gt, mask_pred):
    """
    Calculates the IoU/jaccard index for a pair of masks. 
    
    Logic is implemented to require images of the same size. 

    Parameters
    ----------
    mask_pred: np.ndarray
        Ground truth numpy ndarray image.
    mask_gt: np.ndarray
        Current image numpy ndarray.

    Returns
    -------
    jaccard_index: float
        Calculated Jaccard index/distance.
    """

    if mask_gt.shape != mask_pred.shape:
        raise ValueError(
            f"Shape mismatch: the shape of the ground truth mask {mask_gt.shape} does not match shape of predicted mask {mask_pred.shape}"
        )

    if not mask_gt.any() and not mask_pred.any():
        raise ValueError(
            "Ground truth mask and predicted mask cannot both be entirely zeros"
        )

    mask_gt = _validate_array_and_make_bool(mask_gt)
    mask_pred = _validate_array_and_make_bool(mask_pred)

    return (
        np.logical_and(mask_gt, mask_pred).sum()
        / np.logical_or(mask_gt, mask_pred).sum()
    )
